import gradio as gr
from transformers import pipeline

pipe = pipeline(
    task="image-classification",
    model="microsoft/beit-base-patch16-224"
)

gr.Interface.from_pipeline(
    pipeline=pipe,
    title="22k Image Classification",
    description="Object Recognition using Microsoft BEIT",
    examples=["../ImagesFolder/bayern_bvb.PNG"],
    allow_flagging="never").launch(inbrowser=True)


