# Data

This project uses the provided StackLite corpus archive from:

```text
/Users/mohamedehabelmolla/Downloads/DataSet.zip
```

The archive was extracted into:

- `top_ai_questions.json`
- `top_datascience_questions.json`

Together these files contain 1,500 Stack Exchange question records from the supplied course dataset. They are not empty placeholders or synthetic demo files. The loader in `stacklite_qa/core.py` reads every `top_*_questions.json` file in this folder, so a larger StackLite-6K export can be dropped into the same directory without code changes.

Current local file sizes:

| File | Size |
| --- | ---: |
| `top_ai_questions.json` | 1,418,037 bytes |
| `top_datascience_questions.json` | 1,470,370 bytes |

