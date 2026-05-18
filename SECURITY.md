# Security

## Reporting

If you discover a security vulnerability, please open a private advisory on GitHub or contact the maintainers directly instead of filing a public issue.

## Secrets

Never commit API keys, tokens, or passwords. This project uses environment variables for Weights & Biases (`WANDB_API_KEY`). See `.env.example`.

Do not commit W&B exports, run pickles, or machine-specific paths (for example personal home directories in Docker mount defaults). Use `archive/artifacts/` for local analysis files; those patterns are in `.gitignore`.

If a key was ever committed to version control, rotate it in the provider’s dashboard and consider the old key compromised.
