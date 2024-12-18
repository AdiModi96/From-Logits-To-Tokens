import os
import random

from functions import *

import project_paths as pp
import plotly.graph_objects as go
import gradio as gr

with open(os.path.join(pp.datasets_folder_path, 'words.txt')) as file:
    words = file.read().splitlines()

vocab_size = 50

random.seed(26)
vocab = sorted(random.sample(words, vocab_size))
vocab_frequencies = np.zeros(shape=vocab_size, dtype=np.int16)

np.random.seed(1618)
original_logits = np.random.normal(0, 0.8, size=vocab_size)
original_pmf = convert_logits_to_probabilities(original_logits)


def vocab_dropdown_change(
        word
):
    return vocab_frequencies[vocab.index(word)]

def vocab_frequency_slider_change(
        word,
        frequency,
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        top_k_or_top_p_radio,
        top_k_slider,
        top_p_slider
):
    vocab_frequencies[vocab.index(word)] = frequency
    return update_effective_plots(
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        top_k_or_top_p_radio,
        top_k_slider,
        top_p_slider
    )


def update_original_plots():
    # Building original logits figure
    original_logits_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=original_logits,
            name='Original Logits Distribution',
            marker={
                'color': original_logits,
                'colorscale': 'Burg'
            }
        )
    )

    # Building original PMF figure
    original_pmf_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=original_pmf,
            name='Original Probability Mass Function',
            marker={
                'color': original_pmf,
                'colorscale': 'Burg'
            }
        )
    )

    return original_logits_fig, original_pmf_fig


def update_effective_plots(
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        top_k_or_top_p_radio,
        top_k_slider,
        top_p_slider
):
    effective_logits = original_logits.copy()
    if repetition_penalty_checkbox:
        effective_logits = apply_repetition_penalty(effective_logits, vocab_frequencies, repetition_penalty_slider)
    if frequency_penalty_checkbox:
        effective_logits = apply_frequency_penalty(effective_logits, vocab_frequencies, frequency_penalty_slider)
    if temperature_checkbox:
        effective_logits = apply_temperature(effective_logits, temperature_slider)
    if top_k_or_top_p_radio == 'Top K':
        effective_logits = select_top_k(effective_logits, top_k_slider)
    if top_k_or_top_p_radio == 'Top P':
        effective_logits = select_top_p(effective_logits, top_p_slider)
    effective_pmf = convert_logits_to_probabilities(effective_logits)

    # Building effective logits figure
    effective_logits_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=effective_logits,
            name='Effective Logits Distribution',
            marker={
                'color': effective_logits,
                'colorscale': 'Burg'
            }
        )
    )

    # Building effective PMF figure
    effective_pmf_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=effective_pmf,
            name='Effective Probability Mass Function',
            marker={
                'color': effective_pmf,
                'colorscale': 'Burg'
            }
        )
    )

    return effective_logits_fig, effective_pmf_fig


gr.close_all()
with gr.Blocks() as app:
    gr.Markdown('# From Logits to Tokens')
    with gr.Row():
        with gr.Column(scale=5):
            vocab_dropdown = gr.Dropdown(
                label='Word',
                info='Select a word to change its frequency',
                choices=vocab,
                value=vocab[0],
                interactive=True
            )

            vocab_frequency_slider = gr.Slider(
                label='Word Frequency',
                info='Set the frequency of the above selected word',
                show_label=True,
                minimum=0,
                maximum=10,
                step=1,
                value=0,
                interactive=True
            )
            with gr.Column():
                with gr.Row():
                    repetition_penalty_checkbox = gr.Checkbox(
                        label='Repetition Penalty (RP)',
                        info='Check to enable, uncheck to disable',
                        value=True,
                        interactive=True,
                        scale=1
                    )
                    repetition_penalty_slider = gr.Slider(
                        minimum=0.01,
                        maximum=5,
                        step=0.01,
                        value=1,
                        interactive=True,
                        show_label=False,
                        scale=4
                    )

                with gr.Row():
                    frequency_penalty_checkbox = gr.Checkbox(
                        label='Frequency Penalty (FP)',
                        info='Check to enable, uncheck to disable',
                        value=True,
                        interactive=True,
                        scale=1
                    )
                    frequency_penalty_slider = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=0.01,
                        value=0,
                        interactive=True,
                        show_label=False,
                        scale=4
                    )

                with gr.Row():
                    temperature_checkbox = gr.Checkbox(
                        label='Temperature (T)',
                        info='Check to enable, uncheck to disable',
                        value=True,
                        interactive=True,
                        scale=1
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.01,
                        maximum=25,
                        step=0.01,
                        value=1,
                        interactive=True,
                        show_label=False,
                        scale=4
                    )

                with gr.Row():
                    top_k_or_top_p_radio = gr.Radio(
                        ['Top K', 'Top P'],
                        label='Top K or Top P',
                        info='Select to enable',
                        value='Top K',
                        interactive=True,
                        show_label=True,
                        scale=1
                    )
                    top_k_slider = gr.Slider(
                        label='Top K (top_k)',
                        minimum=0,
                        maximum=len(vocab),
                        step=1,
                        value=len(vocab),
                        interactive=True,
                        show_label=True,
                        scale=4
                    )
                    top_p_slider = gr.Slider(
                        label='Top P (top_p)',
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=1,
                        interactive=True,
                        show_label=True,
                        scale=4
                    )

        with gr.Column(scale=20):
            original_logits_bar_plot = gr.Plot()
            effective_logits_bar_plot = gr.Plot()

        with gr.Column(scale=20):
            original_pmf_bar_plot = gr.Plot()
            effective_pmf_bar_plot = gr.Plot()

    app.load(
        fn=update_original_plots,
        inputs=[],
        outputs=[original_logits_bar_plot, original_pmf_bar_plot]
    )

    app.load(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )

    vocab_dropdown.change(
        fn=vocab_dropdown_change,
        inputs=[vocab_dropdown],
        outputs=[vocab_frequency_slider]
    )

    vocab_frequency_slider.change(
        fn=vocab_frequency_slider_change,
        inputs=[
            vocab_dropdown,
            vocab_frequency_slider,
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )

    repetition_penalty_checkbox.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    repetition_penalty_slider.release(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    frequency_penalty_checkbox.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    frequency_penalty_slider.release(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    temperature_checkbox.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    temperature_slider.release(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    top_k_or_top_p_radio.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    top_k_slider.release(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )
    top_p_slider.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            top_k_or_top_p_radio,
            top_k_slider,
            top_p_slider
        ],
        outputs=[effective_logits_bar_plot, effective_pmf_bar_plot]
    )

app.launch(share=False)
