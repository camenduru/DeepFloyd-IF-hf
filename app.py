#!/usr/bin/env python

import datetime
import hashlib
import json
import os
import random
import tempfile

import gradio as gr
import torch
from huggingface_hub import HfApi

# isort: off
from model import Model
from settings import (
    DEBUG,
    DEFAULT_CUSTOM_TIMESTEPS_1,
    DEFAULT_CUSTOM_TIMESTEPS_2,
    DEFAULT_NUM_IMAGES,
    DEFAULT_NUM_STEPS_3,
    DISABLE_SD_X4_UPSCALER,
    GALLERY_COLUMN_NUM,
    HF_TOKEN,
    MAX_NUM_IMAGES,
    MAX_NUM_STEPS,
    MAX_QUEUE_SIZE,
    MAX_SEED,
    SHOW_ADVANCED_OPTIONS,
    SHOW_CUSTOM_TIMESTEPS_1,
    SHOW_CUSTOM_TIMESTEPS_2,
    SHOW_DEVICE_WARNING,
    SHOW_DUPLICATE_BUTTON,
    SHOW_NUM_IMAGES,
    SHOW_NUM_STEPS_1,
    SHOW_NUM_STEPS_2,
    SHOW_NUM_STEPS_3,
    SHOW_UPSCALE_TO_256_BUTTON,
    UPLOAD_REPO_ID,
    UPLOAD_RESULT_IMAGE,
)
# isort: on

TITLE = '# [DeepFloyd IF](https://github.com/deep-floyd/IF)'
DESCRIPTION = 'The DeepFloyd IF model has been initially released as a non-commercial research-only model. Please make sure you read and abide to the [LICENSE](https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license) before using it.'
DISCLAIMER = 'In this demo, the DeepFloyd team may collect prompts, and user preferences (which of the images the user chose to upscale) for improving future models'
FOOTER = """<div class="footer">
                    <p>Model by <a href="https://huggingface.co/DeepFloyd" style="text-decoration: underline;" target="_blank">DeepFloyd</a> supported by <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">Stability AI</a>
                    </p>
            </div>
            <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a bespoke non-commercial research-only license <a href="https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license" style="text-decoration: underline;" target="_blank">DeepFloyd IF Research License Agreement</a> license. The license forbids you from sharing any content for commercial use, or that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, explicit content and violence. The model was trained on a subset of the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a> and is meant for research purposes. You can read more in the <a href="https://huggingface.co/DeepFloyd/IF-I-IF-v1.0" style="text-decoration: underline;" target="_blank">model card</a></p>
            </div>
        """
if SHOW_DUPLICATE_BUTTON:
    SPACE_ID = os.getenv('SPACE_ID')
    DESCRIPTION += f'\n<p><a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space%20to%20skip%20the%20queue-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p>'

if SHOW_DEVICE_WARNING and not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

model = Model()


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def get_stage2_index(evt: gr.SelectData) -> int:
    return evt.index


def check_if_stage2_selected(index: int) -> None:
    if index == -1:
        raise gr.Error(
            'You need to select the image you would like to upscale from the Stage 1 results by clicking.'
        )


hf_api = HfApi(token=HF_TOKEN)
if UPLOAD_REPO_ID:
    hf_api.create_repo(repo_id=UPLOAD_REPO_ID,
                       private=True,
                       repo_type='dataset',
                       exist_ok=True)


def get_param_file_hash_name(param_filepath: str) -> str:
    if not UPLOAD_REPO_ID:
        return ''
    with open(param_filepath, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    utcnow = datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    return f'{utcnow}-{md5}'


def upload_stage1_result(stage1_param_path: str, stage1_result_path: str,
                         save_name: str) -> None:
    if not UPLOAD_REPO_ID:
        return
    try:
        hf_api.upload_file(path_or_fileobj=stage1_param_path,
                           path_in_repo=f'stage1_params/{save_name}.json',
                           repo_id=UPLOAD_REPO_ID,
                           repo_type='dataset')
        hf_api.upload_file(path_or_fileobj=stage1_result_path,
                           path_in_repo=f'stage1_results/{save_name}.pth',
                           repo_id=UPLOAD_REPO_ID,
                           repo_type='dataset')
    except Exception as e:
        print(e)


def upload_stage2_info(stage1_param_file_hash_name: str,
                       stage2_output_path: str,
                       selected_index_for_upscale: int, seed_2: int,
                       guidance_scale_2: float, custom_timesteps_2: str,
                       num_inference_steps_2: int) -> None:
    if not UPLOAD_REPO_ID:
        return
    if not stage1_param_file_hash_name:
        raise ValueError

    stage2_params = {
        'stage1_param_file_hash_name': stage1_param_file_hash_name,
        'selected_index_for_upscale': selected_index_for_upscale,
        'seed_2': seed_2,
        'guidance_scale_2': guidance_scale_2,
        'custom_timesteps_2': custom_timesteps_2,
        'num_inference_steps_2': num_inference_steps_2,
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as param_file:
        param_file.write(json.dumps(stage2_params))
    stage2_param_file_hash_name = get_param_file_hash_name(param_file.name)
    save_name = f'{stage1_param_file_hash_name}_{stage2_param_file_hash_name}'

    try:
        hf_api.upload_file(path_or_fileobj=param_file.name,
                           path_in_repo=f'stage2_params/{save_name}.json',
                           repo_id=UPLOAD_REPO_ID,
                           repo_type='dataset')
        if UPLOAD_RESULT_IMAGE:
            hf_api.upload_file(path_or_fileobj=stage2_output_path,
                               path_in_repo=f'stage2_results/{save_name}.png',
                               repo_id=UPLOAD_REPO_ID,
                               repo_type='dataset')
    except Exception as e:
        print(e)


def upload_stage2_3_info(stage1_param_file_hash_name: str,
                         stage2_3_output_path: str,
                         selected_index_for_upscale: int, seed_2: int,
                         guidance_scale_2: float, custom_timesteps_2: str,
                         num_inference_steps_2: int, prompt: str,
                         negative_prompt: str, seed_3: int,
                         guidance_scale_3: float,
                         num_inference_steps_3: int) -> None:
    if not UPLOAD_REPO_ID:
        return
    if not stage1_param_file_hash_name:
        raise ValueError

    stage2_3_params = {
        'stage1_param_file_hash_name': stage1_param_file_hash_name,
        'selected_index_for_upscale': selected_index_for_upscale,
        'seed_2': seed_2,
        'guidance_scale_2': guidance_scale_2,
        'custom_timesteps_2': custom_timesteps_2,
        'num_inference_steps_2': num_inference_steps_2,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'seed_3': seed_3,
        'guidance_scale_3': guidance_scale_3,
        'num_inference_steps_3': num_inference_steps_3,
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as param_file:
        param_file.write(json.dumps(stage2_3_params))
    stage2_3_param_file_hash_name = get_param_file_hash_name(param_file.name)
    save_name = f'{stage1_param_file_hash_name}_{stage2_3_param_file_hash_name}'

    try:
        hf_api.upload_file(path_or_fileobj=param_file.name,
                           path_in_repo=f'stage2_3_params/{save_name}.json',
                           repo_id=UPLOAD_REPO_ID,
                           repo_type='dataset')
        if UPLOAD_RESULT_IMAGE:
            hf_api.upload_file(
                path_or_fileobj=stage2_3_output_path,
                path_in_repo=f'stage2_3_results/{save_name}.png',
                repo_id=UPLOAD_REPO_ID,
                repo_type='dataset')
    except Exception as e:
        print(e)


def update_upscale_button(selected_index: int) -> tuple[dict, dict]:
    if selected_index == -1:
        return gr.update(interactive=False), gr.update(interactive=False)
    else:
        return gr.update(interactive=True), gr.update(interactive=True)


def _update_result_view(show_gallery: bool) -> tuple[dict, dict]:
    return gr.update(visible=show_gallery), gr.update(visible=not show_gallery)


def show_gallery_view() -> tuple[dict, dict]:
    return _update_result_view(True)


def show_upscaled_view() -> tuple[dict, dict]:
    return _update_result_view(False)


examples = [
    'high quality dslr photo, a photo product of a lemon inspired by natural and organic materials, wooden accents, intricately decorated with glowing vines of led lights, inspired by baroque luxury',
    'paper quilling, extremely detailed, paper quilling of a nordic mountain landscape, 8k rendering',
    'letters made of candy on a plate that says "diet"',
    'a photo of a violet baseball cap with yellow text: "deep floyd". 50mm lens, photo realism, cine lens. violet baseball cap says "deep floyd". reflections, render. yellow stitch text "deep floyd"',
    'ultra close-up color photo portrait of rainbow owl with deer horns in the woods',
    'a cloth embroidered with the text "laion" and an embroidered cute baby lion face',
    'product image of a crochet Cthulhu the great old one emerging from a spacetime wormhole made of wool',
    'a little green budgie parrot driving small red toy car in new york street, photo',
    'origami dancer in white paper, 3d render, ultra-detailed, on white background, studio shot.',
    'glowing mushrooms in a natural environment with smoke in the frame',
    'a subway train\'s digital sign saying "open source", vsco preset, 35mm photo, film grain, in a dim subway station',
    'a bowl full of few adorable golden doodle puppies, the doodles dusted in powdered sugar and look delicious, bokeh, cannon. professional macro photo, super detailed. cute sweet golden doodle confectionery, baking puppies in powdered sugar in the bowl',
    'a face of a woman made completely out of foliage, twigs, leaves and flowers, side view'
]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Box():
        with gr.Row(elem_id='prompt-container').style(equal_height=True):
            with gr.Column():
                prompt = gr.Text(
                    label='Prompt',
                    show_label=False,
                    max_lines=1,
                    placeholder='Enter your prompt',
                    elem_id='prompt-text-input',
                ).style(container=False)
                negative_prompt = gr.Text(
                    label='Negative prompt',
                    show_label=False,
                    max_lines=1,
                    placeholder='Enter a negative prompt',
                    elem_id='negative-prompt-text-input',
                ).style(container=False)
            generate_button = gr.Button('Generate').style(full_width=False)

        with gr.Column() as gallery_view:
            gallery = gr.Gallery(label='Stage 1 results',
                                 show_label=False,
                                 elem_id='gallery').style(
                                     columns=GALLERY_COLUMN_NUM,
                                     object_fit='contain')
            gr.Markdown('Pick your favorite generation to upscale.')
            with gr.Row():
                upscale_to_256_button = gr.Button(
                    'Upscale to 256px',
                    visible=SHOW_UPSCALE_TO_256_BUTTON
                    or DISABLE_SD_X4_UPSCALER,
                    interactive=False)
                upscale_button = gr.Button('Upscale',
                                           interactive=False,
                                           visible=not DISABLE_SD_X4_UPSCALER)
        with gr.Column(visible=False) as upscale_view:
            result = gr.Image(label='Result',
                              show_label=False,
                              type='filepath',
                              interactive=False,
                              elem_id='upscaled-image').style(height=640)
            back_to_selection_button = gr.Button('Back to selection')

        with gr.Accordion('Advanced options',
                          open=False,
                          visible=SHOW_ADVANCED_OPTIONS):
            with gr.Tabs():
                with gr.Tab(label='Generation'):
                    seed_1 = gr.Slider(label='Seed',
                                       minimum=0,
                                       maximum=MAX_SEED,
                                       step=1,
                                       value=0)
                    randomize_seed_1 = gr.Checkbox(label='Randomize seed',
                                                   value=True)
                    guidance_scale_1 = gr.Slider(label='Guidance scale',
                                                 minimum=1,
                                                 maximum=20,
                                                 step=0.1,
                                                 value=7.0)
                    custom_timesteps_1 = gr.Dropdown(
                        label='Custom timesteps 1',
                        choices=[
                            'none',
                            'fast27',
                            'smart27',
                            'smart50',
                            'smart100',
                            'smart185',
                        ],
                        value=DEFAULT_CUSTOM_TIMESTEPS_1,
                        visible=SHOW_CUSTOM_TIMESTEPS_1)
                    num_inference_steps_1 = gr.Slider(
                        label='Number of inference steps',
                        minimum=1,
                        maximum=MAX_NUM_STEPS,
                        step=1,
                        value=100,
                        visible=SHOW_NUM_STEPS_1)
                    num_images = gr.Slider(label='Number of images',
                                           minimum=1,
                                           maximum=MAX_NUM_IMAGES,
                                           step=1,
                                           value=DEFAULT_NUM_IMAGES,
                                           visible=SHOW_NUM_IMAGES)
                with gr.Tab(label='Super-resolution 1'):
                    seed_2 = gr.Slider(label='Seed',
                                       minimum=0,
                                       maximum=MAX_SEED,
                                       step=1,
                                       value=0)
                    randomize_seed_2 = gr.Checkbox(label='Randomize seed',
                                                   value=True)
                    guidance_scale_2 = gr.Slider(label='Guidance scale',
                                                 minimum=1,
                                                 maximum=20,
                                                 step=0.1,
                                                 value=4.0)
                    custom_timesteps_2 = gr.Dropdown(
                        label='Custom timesteps 2',
                        choices=[
                            'none',
                            'fast27',
                            'smart27',
                            'smart50',
                            'smart100',
                            'smart185',
                        ],
                        value=DEFAULT_CUSTOM_TIMESTEPS_2,
                        visible=SHOW_CUSTOM_TIMESTEPS_2)
                    num_inference_steps_2 = gr.Slider(
                        label='Number of inference steps',
                        minimum=1,
                        maximum=MAX_NUM_STEPS,
                        step=1,
                        value=50,
                        visible=SHOW_NUM_STEPS_2)
                with gr.Tab(label='Super-resolution 2'):
                    seed_3 = gr.Slider(label='Seed',
                                       minimum=0,
                                       maximum=MAX_SEED,
                                       step=1,
                                       value=0)
                    randomize_seed_3 = gr.Checkbox(label='Randomize seed',
                                                   value=True)
                    guidance_scale_3 = gr.Slider(label='Guidance scale',
                                                 minimum=1,
                                                 maximum=20,
                                                 step=0.1,
                                                 value=9.0)
                    num_inference_steps_3 = gr.Slider(
                        label='Number of inference steps',
                        minimum=1,
                        maximum=MAX_NUM_STEPS,
                        step=1,
                        value=DEFAULT_NUM_STEPS_3,
                        visible=SHOW_NUM_STEPS_3)

    gr.Examples(examples=examples, inputs=prompt, examples_per_page=4)

    with gr.Box(visible=DEBUG):
        with gr.Row():
            with gr.Accordion(label='Hidden params'):
                stage1_param_path = gr.Text(label='Stage 1 param path')
                stage1_result_path = gr.Text(label='Stage 1 result path')
                stage1_param_file_hash_name = gr.Text(
                    label='Stage 1 param file hash name')
                selected_index_for_stage2 = gr.Number(
                    label='Selected index for Stage 2', value=-1, precision=0)
    gr.Markdown(DISCLAIMER)
    gr.HTML(FOOTER)
    stage1_inputs = [
        prompt,
        negative_prompt,
        seed_1,
        num_images,
        guidance_scale_1,
        custom_timesteps_1,
        num_inference_steps_1,
    ]
    stage1_outputs = [
        gallery,
        stage1_param_path,
        stage1_result_path,
    ]

    prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed_1, randomize_seed_1],
        outputs=seed_1,
        queue=False,
    ).then(
        fn=lambda: -1,
        outputs=selected_index_for_stage2,
        queue=False,
    ).then(
        fn=show_gallery_view,
        outputs=[
            gallery_view,
            upscale_view,
        ],
        queue=False,
    ).then(
        fn=update_upscale_button,
        inputs=selected_index_for_stage2,
        outputs=[
            upscale_button,
            upscale_to_256_button,
        ],
        queue=False,
    ).then(
        fn=model.run_stage1,
        inputs=stage1_inputs,
        outputs=stage1_outputs,
    ).success(
        fn=get_param_file_hash_name,
        inputs=stage1_param_path,
        outputs=stage1_param_file_hash_name,
        queue=False,
    ).then(
        fn=upload_stage1_result,
        inputs=[
            stage1_param_path,
            stage1_result_path,
            stage1_param_file_hash_name,
        ],
        queue=False,
    )

    negative_prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed_1, randomize_seed_1],
        outputs=seed_1,
        queue=False,
    ).then(
        fn=lambda: -1,
        outputs=selected_index_for_stage2,
        queue=False,
    ).then(
        fn=show_gallery_view,
        outputs=[
            gallery_view,
            upscale_view,
        ],
        queue=False,
    ).then(
        fn=update_upscale_button,
        inputs=selected_index_for_stage2,
        outputs=[
            upscale_button,
            upscale_to_256_button,
        ],
        queue=False,
    ).then(
        fn=model.run_stage1,
        inputs=stage1_inputs,
        outputs=stage1_outputs,
    ).success(
        fn=get_param_file_hash_name,
        inputs=stage1_param_path,
        outputs=stage1_param_file_hash_name,
        queue=False,
    ).then(
        fn=upload_stage1_result,
        inputs=[
            stage1_param_path,
            stage1_result_path,
            stage1_param_file_hash_name,
        ],
        queue=False,
    )

    generate_button.click(
        fn=randomize_seed_fn,
        inputs=[seed_1, randomize_seed_1],
        outputs=seed_1,
        queue=False,
    ).then(
        fn=lambda: -1,
        outputs=selected_index_for_stage2,
        queue=False,
    ).then(
        fn=show_gallery_view,
        outputs=[
            gallery_view,
            upscale_view,
        ],
        queue=False,
    ).then(
        fn=update_upscale_button,
        inputs=selected_index_for_stage2,
        outputs=[
            upscale_button,
            upscale_to_256_button,
        ],
        queue=False,
    ).then(
        fn=model.run_stage1,
        inputs=stage1_inputs,
        outputs=stage1_outputs,
        api_name='generate64',
    ).success(
        fn=get_param_file_hash_name,
        inputs=stage1_param_path,
        outputs=stage1_param_file_hash_name,
        queue=False,
    ).then(
        fn=upload_stage1_result,
        inputs=[
            stage1_param_path,
            stage1_result_path,
            stage1_param_file_hash_name,
        ],
        queue=False,
    )

    gallery.select(
        fn=get_stage2_index,
        outputs=selected_index_for_stage2,
        queue=False,
    )

    selected_index_for_stage2.change(
        fn=update_upscale_button,
        inputs=selected_index_for_stage2,
        outputs=[
            upscale_button,
            upscale_to_256_button,
        ],
        queue=False,
    )

    stage2_inputs = [
        stage1_result_path,
        selected_index_for_stage2,
        seed_2,
        guidance_scale_2,
        custom_timesteps_2,
        num_inference_steps_2,
    ]

    upscale_to_256_button.click(
        fn=check_if_stage2_selected,
        inputs=selected_index_for_stage2,
        queue=False,
    ).then(
        fn=randomize_seed_fn,
        inputs=[seed_2, randomize_seed_2],
        outputs=seed_2,
        queue=False,
    ).then(
        fn=show_upscaled_view,
        outputs=[
            gallery_view,
            upscale_view,
        ],
        queue=False,
    ).then(
        fn=model.run_stage2,
        inputs=stage2_inputs,
        outputs=result,
        api_name='upscale256',
    ).success(
        fn=upload_stage2_info,
        inputs=[
            stage1_param_file_hash_name,
            result,
            selected_index_for_stage2,
            seed_2,
            guidance_scale_2,
            custom_timesteps_2,
            num_inference_steps_2,
        ],
        queue=False,
    )

    stage2_3_inputs = [
        stage1_result_path,
        selected_index_for_stage2,
        seed_2,
        guidance_scale_2,
        custom_timesteps_2,
        num_inference_steps_2,
        prompt,
        negative_prompt,
        seed_3,
        guidance_scale_3,
        num_inference_steps_3,
    ]

    upscale_button.click(
        fn=check_if_stage2_selected,
        inputs=selected_index_for_stage2,
        queue=False,
    ).then(
        fn=randomize_seed_fn,
        inputs=[seed_2, randomize_seed_2],
        outputs=seed_2,
        queue=False,
    ).then(
        fn=randomize_seed_fn,
        inputs=[seed_3, randomize_seed_3],
        outputs=seed_3,
        queue=False,
    ).then(
        fn=show_upscaled_view,
        outputs=[
            gallery_view,
            upscale_view,
        ],
        queue=False,
    ).then(
        fn=model.run_stage2_3,
        inputs=stage2_3_inputs,
        outputs=result,
        api_name='upscale1024',
    ).success(
        fn=upload_stage2_3_info,
        inputs=[
            stage1_param_file_hash_name,
            result,
            selected_index_for_stage2,
            seed_2,
            guidance_scale_2,
            custom_timesteps_2,
            num_inference_steps_2,
            prompt,
            negative_prompt,
            seed_3,
            guidance_scale_3,
            num_inference_steps_3,
        ],
        queue=False,
    )

    back_to_selection_button.click(
        fn=show_gallery_view,
        outputs=[
            gallery_view,
            upscale_view,
        ],
        queue=False,
    )

demo.queue(api_open=False, max_size=MAX_QUEUE_SIZE).launch()
