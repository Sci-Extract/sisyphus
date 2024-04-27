# Chain elements signature (input, output)

| Elements/IO type | Input | Output |
| ---- | ---- | ---- |
| Parser | .html | list[Document] |
| Filter | Document, file name [,query] | bool |
| Extractor | Document | Pydantic |
| Validator | list[Pydantic] | list[Pydantic] |
| Writer | list[Pydantic] | None |

:shipit:
> [!NOTE]
> To resolve the demonstrative pronouns in extraction results, validator was used on the scope of an article.