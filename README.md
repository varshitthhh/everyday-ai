# Gemma 4 Multimodal Workflow

Run a local multimodal workflow with Gemma 4 and Ollama. Given a directory of
images, it produces structured memories for each image and a summary of the
collection.

The workflow combines visual analysis with basic EXIF metadata such as the
filename, dimensions, capture time, and GPS coordinates when available. It is
implemented in Python and can be run from the command line or explored in the
included notebook.

## Workflow

```text
images -> prepare and read metadata -> analyze each image
       -> synthesize the collection -> JSON output
```

Python controls the sequence and validates Gemma's structured responses. The
model does not call tools or access the internet.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- A vision-capable Gemma 4 model available to Ollama

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull gemma4:e4b-128k
```

Run the workflow against a directory of `.jpg`, `.jpeg`, `.png`, or `.webp`
files:

```powershell
python multimodal_workflow.py "C:\path\to\your\images"
```

The result is written to `outputs/trip_memory.json`. Input images and generated
outputs are ignored by Git, so keep private images outside the repository.

To change the model or output path:

```powershell
python multimodal_workflow.py "C:\path\to\your\images" `
  --model gemma4:e4b-128k `
  --output outputs/my-trip.json
```

## Notebook

Install Jupyter and open the included notebook:

```powershell
python -m pip install jupyter
jupyter notebook notebooks/multimodal_workflow.ipynb
```

Set `PHOTO_DIR` to a local image directory. The notebook walks through image
preparation, single-image analysis, collection-wide processing, and final
synthesis.

## Results

The output contains the original metadata, one validated analysis per image,
and a validated collection summary:

```json
{
  "photo_memories": [
    {
      "photo_id": "IMG_0001.jpg",
      "metadata": {"filename": "IMG_0001.jpg", "width": 4032, "height": 3024},
      "analysis": {"scene_summary": "...", "confidence": 0.9}
    }
  ],
  "trip_memory": {
    "narrative_summary": "...",
    "recurring_themes": [],
    "memorable_moments": [],
    "evidence_photo_ids": []
  }
}
```

Evidence references use the supplied filenames. The workflow rejects a
synthesis that cites an image that was not provided.

## Model selection

The default model is `gemma4:e4b-128k`. Override it with `--model` or the
`GEMMA_MODEL` environment variable:

```powershell
$env:GEMMA_MODEL = "your-local-gemma-model"
python multimodal_workflow.py "C:\path\to\your\images"
```

The selected model must support image input. If it responds without using the
image, use a vision-capable variant with its matching multimodal projector.

## Privacy

Images are sent to the Ollama endpoint used by your local client. Verify that
Ollama is running locally if local-only processing is important to you. EXIF
GPS data can be sensitive, and model-generated descriptions should be treated
as interpretations rather than verified facts.
