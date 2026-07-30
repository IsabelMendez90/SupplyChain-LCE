# OpenRouter connection and audit guide

## Streamlit secret

Configure the key in Streamlit Community Cloud at the top level:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

The repository contains only `.streamlit/secrets.toml.example`. It does not
override the secret stored by Streamlit Cloud.

## Router

The application sends:

```python
model="openrouter/free"
```

No individual provider model is forced. OpenRouter selects an available free
model and the response audit records the realized model.

## Audit statuses

- `accepted`: the model returned text and it passed deterministic grounding.
- `rejected`: text was returned but contradicted, omitted, or exceeded the
  frozen evidence constraints.
- `empty_response`: OpenRouter returned a model response without reader-facing
  text. The actual model and finish reason remain recorded.
- `api_error`: the request raised a genuine transport, authentication, quota,
  provider, or HTTP exception. The audit includes sanitized diagnostic fields.

All non-accepted outcomes use the deterministic explanation. Fuzzy scores and
rankings never depend on OpenRouter availability.

## Deployment check

After replacing the repository contents:

1. reboot the Streamlit app;
2. confirm that the explanation layer reports the key as detected;
3. click `Analyze`;
4. inspect one audit entry if a fallback occurs;
5. use the `status_code`, `code`, and sanitized `message` fields to distinguish
   authentication, rate-limit, provider-capacity, and transport failures.

