
__all__ = ['learn','classify_image','categories','image','label','examples','intf']

from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')


categories = ('Car', 'Bike')
def classify_image(img):
    is_car,_,probs = learn.predict(PILImage.create(img))
    return dict(zip(categories, map(float,probs))) #gradio only supports floats and it doesn't handle PyTorch tensors

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['volkswagen.jpg','motorbike.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)