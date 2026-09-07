# Deploying the Space

Everything a Hugging Face Space needs is already in this repository: `app.py`,
`requirements.txt` and `packages.txt` at the root, plus `space/README.md`, which
carries the YAML header Spaces read for their title, SDK and hardware.

Hosting a Gradio Space requires a Hugging Face PRO account — free `cpu-basic` is
open to static Spaces only, and this demo needs a Python backend to run the models.
With PRO:

```bash
hf repos create <user>/piano-music-transformer-demo --type space --sdk gradio --public
hf upload <user>/piano-music-transformer-demo space/README.md README.md --type space
hf upload <user>/piano-music-transformer-demo app.py app.py --type space
hf upload <user>/piano-music-transformer-demo requirements.txt requirements.txt --type space
hf upload <user>/piano-music-transformer-demo packages.txt packages.txt --type space
```

`requirements.txt` installs this project straight from GitHub, so the Space tracks
`main` and the two cannot drift apart. On first boot `app.py` downloads the
published weights and a soundfont; nothing else needs configuring.

Expect the first build to take several minutes — it compiles nothing, but PyTorch
is a large download.

## Without PRO

The same interface runs locally with one command, and locally it is better: it can
also continue a real MAESTRO excerpt, which the hosted version cannot, because
publishing the tokenised dataset would redistribute CC BY-NC-SA data.

```bash
uv run --extra demo python app.py
```
