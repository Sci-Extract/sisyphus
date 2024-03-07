# The caveats is that the name inside one node can't be duplicated
import json
import random
import subprocess
from typing import List
from collections import namedtuple

import streamlit as st
import pandas as pd
from graphviz import Digraph
from PIL import Image
from io import BytesIO
from pydantic import create_model, BaseModel, Field

import logging
logging.basicConfig(level=10)


DictWrapper = namedtuple("DictWrapper", ['name', 'type', 'description', 'is_array', 'field_of', 'depth'])
PrimalFieldWrapper = namedtuple("PrimalFieldWrapper", ['name', 'type', 'description', 'is_array'])

class DynamicPydantic:
    def __init__(self, base_model: BaseModel):
        self.base_model = base_model
        self.field_of = "Compound"
    
    def add_field(self, primal_field_wrapper: PrimalFieldWrapper, dict_wrappers: List[DictWrapper] = None):
        if primal_field_wrapper.type == "object":
            type_annoatation = self.build_block(dict_wrappers)
        else:
            type_annoatation = primal_field_wrapper.type
        if primal_field_wrapper.is_array:
            type_annoatation = List[type_annoatation]
    
        self.base_model = create_model(
            self.field_of,
            __base__=self.base_model,
            **{
                primal_field_wrapper.name: (type_annoatation, Field(None, description=primal_field_wrapper.description))
            }
        )

    def build_block(self, dict_wrappers: List[DictWrapper]) -> BaseModel:
        # TODO: can apply recursive logic to this, try it out!
        # ^-^, I just realize that if I keep this code, there gonna to be have a greate chance that I will forget what the hell does this all about in the future.
        """only be called when user want to use nested structure"""
        annoatation_dict = {}
        #  dict_wrapper = [{"name": "unit", "type": "str", "description": "unit of property", "depth": 1}, {"name": "odd", "type": "object", "description": "odd", "depth": 1}, {"field_of": "odd", "name": "boo", "type": "str", "description": "boo", "depth": 2}]
        df = pd.DataFrame(dict_wrappers)
        grouped = df.groupby('field_of')
        df_for_each = [df for _, df in grouped] # collapse primal dataframe to small ones based on field_of
        df_sorted: List[pd.DataFrame] = sorted(df_for_each, key=lambda x: x.depth.iloc[0], reverse=True) # sort df by depth
        for df in df_sorted:
            field_of: str = df.field_of.iloc[0]
            field_definitions = {}
            for row in df.itertuples(index=False):
                if row.type == 'object':
                    type_ = annoatation_dict[row.name]
                else:
                    type_ = row.type
                if row.is_array:
                    type_ = List[type_]
                field_definitions.update(
                    {row.name: (type_, Field(None, description=row.description))}
                )
            annoatation_dict[field_of] = create_model(
                field_of.capitalize(),
                **field_definitions
            )
        field_of = df_sorted[-1]["field_of"].iloc[0] # the outermost field name
        return annoatation_dict[field_of]

annotation_dict = {}
def compose_node_annotation(node):
    field_defs = {}
    for child_node in node:
        field_defs.update({child_node.name: child_node.type_})
    annotation_dict.update()

def compose_recursively(node):
    for child_node in node:
        if child_node.is_nested:
            compose_recursively(child_node)
    compose_node_annotation(node)

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

class PydanticNode:
    def __init__(self, name, parent_node: "PydanticNode"):
        self.name = name
        self.parent_node = parent_node
        self.child_node = []
        
    def diagram(self, indent=0):
        indent_char = "*" * indent
        st.markdown(f"{indent_char}{self.name}")
        for node in self.child_node:

            depth = st.session_state.registry[node.name]
            node.diagram(depth)

    def print_tree(self, level=0, prefix="", last_sibling=True):
        print("  " * level + prefix + ("└─ " if last_sibling else "├─ ") + str(self.name))
        for i, child in enumerate(self.child_node):
            is_last = i == len(self.child_node) - 1
            child.print_tree(level + 1, "    " * (level + 1) + (" " if last_sibling or is_last else "│"), is_last)

class Nested(PydanticNode):
    def __init__(self, name, parent_node):
        super().__init__(name, parent_node)
        self.child_node = []

    def add_child(self, node):
        self.child_node.append(node)
    
    
class Root(Nested):
    def __init__(self, name):
        super().__init__(name, parent_node=None)
        self.child_node = []

def update_depth(registry, node):
    depth_of_parent = registry.get(node.parent_node.name)
    depth = depth_of_parent + 1
    registry[node.name] = depth # update
    return depth

# Recursive function to create edges between nodes
def add_edges(node, graph, edges, name_list):
    for child in node.child_node:
        if (name:=child.name) not in name_list:
            name_list.append(name)
        else:
            name = node.name + '_' + child.name
        
        graph.node(name)
        edges.append((node.name, name))
        add_edges(child, graph, edges, name_list)

# Function to display the tree using graphviz
def display_tree(root):
    graph = Digraph(comment="Tree Structure", format='png')
    graph.node(root.name)
    edges = []
    name_list = []
    add_edges(root, graph, edges, name_list)
    graph.edges(edges)

    # Render the graph using graphviz
    image_bytes = graph.pipe(format='png')
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption="Tree Structure", use_column_width="auto", width=100)

# instantiate session variables
if "pydantic_model" not in st.session_state:
    st.session_state.pydantic_model = create_model("Unit", __doc__="unit of extracted instance")
if "field_definitions" not in st.session_state:
    st.session_state.field_definitions = {"primal_field_wrapper": None, "dict_wrappers": []}
if "registry" not in st.session_state:
    st.session_state.registry = {}
if "node" not in st.session_state:
    st.session_state.node = {}
    st.session_state.node_names = []
if "node_name" not in st.session_state:
    st.session_state.node_name = ""
    st.session_state.select = ""
if "root_among_sessions" not in st.session_state:
    st.session_state.root_among_sessions = {}
if "fields" not in st.session_state:
    st.session_state.fields = []

# basic config
st.set_page_config(page_title="Data Model", page_icon="📦")
st.title("Building :blue[Data] Model")
st.subheader("""supported by [pydantic](https://docs.pydantic.dev/latest/)"""
             , divider='rainbow')
with st.sidebar:
    st.write("")
    st.markdown("💕Special thanks to pydantic")

def trigger_function():
    if not st.session_state.node:
        is_nested = True
        name = st.session_state.node_name
        st.session_state.node_names.append(name)
        if st.session_state.type == "object":
            node = Root(name)
        else:
            node = PydanticNode(name, parent_node=None)
            is_nested = False
            st.warning("Note that you are just creating a property without nested fields", icon="⚠️")

        st.session_state.node[name] = node
        st.session_state.fields.append(node)
        st.session_state.registry[name] = 0
        collect_info(node=node, mode="root")
        # once user define a not nested model, some action to take...

    else:
        name = st.session_state.node_name
        logging.info(name)
        parent_node = st.session_state.node[st.session_state.select]
        if st.session_state.type == "object":
            node = Nested(name=name, parent_node=parent_node)
        else:
            node = PydanticNode(name=name, parent_node=parent_node)
        parent_node.add_child(node)
        st.session_state.node_names.append(name)
        st.session_state.node[name] = node
        update_depth(registry=st.session_state.registry, node=node)
        collect_info(node=node)

def collect_info(node, mode=None):
    wrapper = PrimalFieldWrapper(
        st.session_state.node_name,
        st.session_state.type,
        st.session_state.description,
        st.session_state.is_array
    )
    if mode == "root":
        st.session_state.field_definitions["primal_field_wrapper"] = wrapper
        return
    depth = st.session_state.registry[node.name]
    field_of = node.parent_node.name
    dict_wrapper = DictWrapper(*wrapper, field_of=field_of, depth=depth)
    st.session_state.field_definitions["dict_wrappers"].append(dict_wrapper)



st.markdown("#### Property Definition")

st.write("")
# create node
with st.form("form", clear_on_submit=True):
    st.markdown("##### Define fields")
    st.markdown("- First define the root (property), then define rest of fields if nested")
    options = [
        node.name for node in st.session_state.node.values() if hasattr(node, "add_child")
    ] if st.session_state.node else []
    select_node = st.selectbox(
        "Select parent field",
        options=options,
        key="select")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    with c1:
        name = st.text_input("Name of the field", key="node_name")
    with c2:
        type_ = st.selectbox(
            "Type of the field",
            options=["int", "float", "str", "bool", "object"],
            index=None,
            key="type",
            help="the former 4 types is basic types, select object when needs nested model"
            )
    with c3:
        description = st.text_input("Description of the field", key="description")
    with c4:
        is_array = st.selectbox("Field is array", options=[False, True], key="is_array")
    

    button = st.form_submit_button("create", on_click=trigger_function)


def clear_sesssion_callback():
    dm = DynamicPydantic(base_model=st.session_state.pydantic_model)
    dm.add_field(**st.session_state.field_definitions) # update field
    st.session_state.pydantic_model = dm.base_model

    # root_name = st.session_state.node_names[0]
    # root = st.session_state.node[root_name]
    # st.session_state.root_among_sessions.update({root_name: root})

    st.session_state.field_definitions = {"primal_field_wrapper": None, "dict_wrappers": []}
    st.session_state.registry = {}
    st.session_state.node = {}
    st.session_state.node_names = []
    st.session_state.node_name = ""
    st.session_state.select = ""
    
st.markdown("Click below once you have defined a property")
done_button = st.button("Done", on_click=clear_sesssion_callback)


st.markdown("---")
st.markdown("_Your defined structure is alike_...")
if st.session_state.node:
    display_tree(root=st.session_state.node[st.session_state.node_names[0]])
st.markdown("---")

def hard_code_pydantic():
    json_schema_file = "gen_schema.json"
    pydantic_code = "gen_pydantic.py"
    command = f"datamodel-codegen --input {json_schema_file} --output {pydantic_code}"
    with open(json_schema_file, 'w', encoding='utf-8') as file:
        file.write(json.dumps(st.session_state.pydantic_model.model_json_schema(), indent=2))
    subprocess.run(command, shell=True, check=True)
    st.write("Succeed generate pydantic model")

def finish_call_back():
    unit_model = st.session_state.pydantic_model
    model = create_model(
        "Compounds",
        __doc__="list of extracted units",
        compounds=(List[unit_model], None)
    )
    st.session_state.pydantic_model = model # update
    compound = Nested('compound', None)
    for child in st.session_state.fields:
        compound.add_child(child)
    st.markdown("_The final data structure was alike_...")
    display_tree(root=compound)
    st.markdown("")
    hard_code_pydantic()

Finish = st.button("Finish", on_click=finish_call_back)

# init_wrapper = input_schema(0)
# root = PydanticNode(name="root", parent_node=None)
# first_node = PydanticNode(name=init_wrapper.name, parent_node=root)








# # streamlit construct flow
# dp = DynamicPydantic(base_model=st.session_state.pydantic_model)
# # user input logic
# #######
# name, type_, description, is_array, dict_wrappers = 1, 2,3 ,4, 5
# #######
# dp.add_field(name, type_, description, is_array, dict_wrappers) # add field to base model
# st.session_state.pydantic_model = dp.base_model # update
# st.json(
#     json.dumps(dp.base_model.model_json_schema())
# )


# dict_wrapper = [{"field_of": "test", "name": "unit", "type": "str", "description": "unit of property", "is_array": False, "depth": 1}, {"field_of": "test", "name": "odd", "type": "object", "description": "odd","is_array": True, "depth": 1}, {"field_of": "odd", "name": "boo", "type": "object","is_array": False, "description": "boofoo", "depth": 2}, {"field_of": "boo", "name": "lol", "type": "str", "description": "lolol","is_array": False, "depth":3}]
# dp = DynamicPydantic(base_model=st.session_state.pydantic_model)
# annoatation = dp.build_block(dict_wrappers=dict_wrapper)
# Model = create_model("Model", test=(annoatation, ...))
