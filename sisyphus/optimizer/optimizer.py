import logging

import dspy
from dspy.evaluate import normalize_text

from .intention import guess_intention
from .bootstrap import bootstrapper_agent
from .evaluator import exec_eval_parallel, prioritized_fields_default
from .reflexion import reflexion
from .proposal import propose_agent
from .utils import dump_json


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('optimize.log', encoding='utf-8', mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)


class Compiler:
    def __init__(
            self,
            bootstrapped_temp: int = 1.4,
            bootstrapped_nums: int = 4,
            propose_temp: float = 0.7,
            propose_nums: int = 5,
            num_threads: int = 10,
            threshold: float = 1.0,
            iterations: int = 3,
            prompt_model: dspy.LM = None,
            bootstrapper=bootstrapper_agent,
            exec_eval_agent=exec_eval_parallel,
            prioritize_field_func=prioritized_fields_default,
            reflexion=reflexion,
            propose_agent=propose_agent,
            return_k: int = 3
    ):
        """assume program output only have one field"""

        self.guess_intention = guess_intention
        self.bootstrapped_temp = bootstrapped_temp
        self.bootstrapped_nums = bootstrapped_nums
        self.propose_temp = propose_temp
        self.propose_nums = propose_nums
        self.bootstrapper = bootstrapper
        self.num_threads = num_threads
        self.threshold = threshold
        self.iterations = iterations
        self.prompt_model = prompt_model
        self.exec_eval_agent = exec_eval_agent
        self.prioritize_field_func = prioritize_field_func
        self.refelxion = reflexion
        self.propose_agent = propose_agent
        self.return_k = return_k

        self.last_key = None

    def compile(self, program, train_set, val_set, seed=22):
        assert len(program.predictors()) == 1, 'Only one predictor is allowed for now.'
        predictor = program.predictors()[0]
        initial_instruction = self._get_signature(predictor).instructions
        *_, self.last_key = self._get_signature(predictor).fields.keys()

        if self.prompt_model: # turn this into wrapper
            with dspy.context(lm=self.prompt_model):
                program_description = guess_intention(train_set, seed=seed)
        else:
            program_description = guess_intention(train_set, seed=seed)
        logger.debug('Program description: %s', program_description)

        # create few instructions
        if self.prompt_model:
            with dspy.context(lm=self.prompt_model):
                candidates = self.deduplicate(self.bootstrapper(program_description, self.bootstrapped_temp, self.bootstrapped_nums))
        else:
            candidates = self.deduplicate(self.bootstrapper(program_description, self.bootstrapped_temp, self.bootstrapped_nums))
        candidates.append(initial_instruction)
        logger.debug('Candidates instructions:\n%s', '\n'.join(candidates))

        program_copy = program.deepcopy()
        predictor_copy = program_copy.predictors()[0]
        evaluated = {}
        new_candidates = candidates

        for i in range(self.iterations):
            logger.debug('======Iteration: %s======', i+1)
            # evaluate candidates
            for instruction in new_candidates:
                updated_signature = (
                    self._get_signature(predictor_copy)
                    .with_instructions(instruction)
                )
                self._set_signature(predictor_copy, updated_signature)

                evaluated_attemps, score = self.exec_eval_agent(
                    program_copy,
                    self.last_key,
                    train_set,
                    self.num_threads,
                    self.prioritize_field_func
                )
                evaluated[instruction] = {
                    'program': program_copy.deepcopy(),
                    'instruction': instruction,
                    'attempts': evaluated_attemps,
                    'iteration': i,
                    'score': score
                }
                logger.debug('Instruction: %s, Score: %s', instruction, score)
                if score >= self.threshold:
                    logger.debug('Threshold reached, stop the iteration')
                    return self._return_top_k(evaluated, val_set)

            if i == self.iterations - 1:
                logger.debug('Last iteration reached, stop the iteration')
                break

            # reflection, using the best candidate
            evaluated_i = {k: v for k, v in evaluated.items() if v['iteration'] == i}
            sorted_candidates = sorted(evaluated_i.values(), key=lambda x: x['score'], reverse=True)
            logger.debug('Using best candidate: %s', sorted_candidates[0]['instruction'])
            if self.prompt_model:
                with dspy.context(lm=self.prompt_model):
                    pattern_with_suggestion = self.refelxion(
                        task_description=program_description,
                        evaluates=sorted_candidates[0]['attempts'],
                        extract_instruction=sorted_candidates[0]['instruction'],
                        use_passed_num=3,
                        seed=seed
                    )
            else:
                pattern_with_suggestion = self.refelxion(
                    task_description=program_description,
                    evaluates=sorted_candidates[0]['attempts'],
                    extract_instruction=sorted_candidates[0]['instruction'],
                    use_passed_num=3,
                    seed=seed
                )
            logger.debug('Reflexion:\nPatterns: %s\nSuggestions: %s', pattern_with_suggestion.patterns, pattern_with_suggestion.suggestions)

            # propose new instructions
            sorted_candidates_all = sorted(evaluated.values(), key=lambda x: x['score'], reverse=True)
            attempted_instructions = [
                {
                    'intstruction': candidate['instruction'],
                    'score': candidate['score']
                }
                for candidate in sorted_candidates_all[::-1][:5] # top-5 as reference insturctions
            ]
            if self.prompt_model:
                with dspy.context(lm=self.prompt_model):
                    new_candidates = self.propose_agent(
                        program_description,
                        dump_json(attempted_instructions),
                        pattern_with_suggestion.patterns,
                        pattern_with_suggestion.suggestions,
                        self.propose_temp,
                        self.propose_nums
                    )
            else:
                new_candidates = self.propose_agent(
                    program_description,
                    dump_json(attempted_instructions),
                    pattern_with_suggestion.patterns,
                    pattern_with_suggestion.suggestions,
                    self.propose_temp,
                    self.propose_nums
                )
            new_candidates = self.deduplicate(new_candidates)
            self.propose_temp += 0.1
            logger.debug('New candidates instructions:\n%s', '\n'.join(new_candidates))
        
        return self._return_top_k(evaluated, val_set)

    def _get_signature(self, predictor):
        if hasattr(predictor, "extended_signature"):
            return predictor.extended_signature
        elif hasattr(predictor, "signature"):
            return predictor.signature

    def _set_signature(self, predictor, updated_signature):
        if hasattr(predictor, "extended_signature"):
            predictor.extended_signature = updated_signature
        elif hasattr(predictor, "signature"):
            predictor.signature = updated_signature
    
    def deduplicate(self, sequence):
        seen = set()
        seen_add = seen.add
        return [normalize_text(i) for i in sequence if not (normalize_text(i) in seen or seen_add(normalize_text(i)))]
    
    def _return_top_k(self, evaluated, val_set):
        """return the best program based on the evaluation on the validation set"""
        sorted_candidates = sorted(evaluated.values(), key=lambda x: x['score'], reverse=True)
        top_k_score = [candidates['score'] for candidates in sorted_candidates[:self.return_k]]
        top_k = [candidates['program'] for candidates in sorted_candidates[:self.return_k]]
        scores = list(map(self._eval_on_dev, top_k, [val_set]*self.return_k))
        index = scores.index(max(scores))
        best_program = top_k[index]
        predictor = best_program.predictors()[0]
        instruction = self._get_signature(predictor).instructions
        dev_score = max(scores)
        train_score = top_k_score[index]
        logger.debug('Best instruction: %s', instruction)
        logger.debug('train score: %s', train_score)
        logger.debug('dev score: %s', dev_score)
        return best_program
    
    def _eval_on_dev(self, program, val_set):
        _, score = self.exec_eval_agent(
            program,
            self.last_key,
            val_set,
            self.num_threads,
            self.prioritize_field_func
        )
        return score


# TODO: wrapped the context manager for prompt model
