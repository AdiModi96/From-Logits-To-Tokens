import os
import random
from functions import *
import project_paths as pp
import plotly.graph_objects as go
import gradio as gr

with open(os.path.join(pp.datasets_folder_path, 'words.txt')) as file:
    words = file.read().splitlines()

vocab_size = 25

random.seed(vocab_size)
vocab = sorted(random.sample(words, vocab_size))
vocab_frequencies = np.zeros(shape=vocab_size, dtype=np.int16)

np.random.seed(vocab_size)
original_logits = np.random.normal(0, 0.8, size=vocab_size)
effective_logits = original_logits.copy()
original_pmf = convert_logits_to_probabilities(original_logits)
effective_pmf = original_pmf.copy()


def vocab_dropdown_change(word):
    global vocab, vocab_frequencies, original_logits, effective_logits, original_pmf, effective_pmf
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
    global vocab, vocab_frequencies, original_logits, effective_logits, original_pmf, effective_pmf
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
    global vocab, vocab_frequencies, original_logits, effective_logits, original_pmf, effective_pmf

    # Building original logits figure
    original_logits_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=original_logits,
            marker={
                'color': original_logits,
                'colorscale': 'Burg'
            }
        )
    )
    original_logits_fig.update_layout(
        title_text='Original Logits Distribution',
        title_x=0.5,
        yaxis={'title': {'text': 'Logits'}}
    )

    # Building original PMF figure
    original_pmf_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=original_pmf,
            marker={
                'color': original_pmf,
                'colorscale': 'Burg'
            }
        )
    )
    original_pmf_fig.update_layout(
        title_text='Original Probability Mass Function',
        title_x=0.5,
        yaxis={'title': {'text': 'Probability'}}
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
    global vocab, vocab_frequencies, original_logits, effective_logits, original_pmf, effective_pmf
    effective_logits = original_logits.copy()
    if repetition_penalty_checkbox:
        effective_logits = apply_repetition_penalty(effective_logits, vocab_frequencies, repetition_penalty_slider)
    if frequency_penalty_checkbox:
        effective_logits = apply_frequency_penalty(effective_logits, vocab_frequencies, frequency_penalty_slider)
    if temperature_checkbox:
        effective_logits = apply_temperature(effective_logits, temperature_slider)
    if top_k_or_top_p_radio == 'Top K':
        effective_logits = select_top_k(effective_logits, top_k_slider)
    if top_k_or_top_p_radio == 'Top P (Nucleus)':
        effective_logits = select_top_p(effective_logits, top_p_slider)
    effective_pmf = convert_logits_to_probabilities(effective_logits)

    # Building effective logits figure
    effective_logits_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=effective_logits,
            marker={
                'color': effective_logits,
                'colorscale': 'Burg'
            }
        )
    )
    effective_logits_fig.update_layout(
        title_text='Effective Logits Distribution',
        title_x=0.5,
        yaxis={'title': {'text': 'Logits'}}
    )

    # Building effective PMF figure
    effective_pmf_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=effective_pmf,
            marker={
                'color': effective_pmf,
                'colorscale': 'Burg'
            }
        )
    )
    effective_pmf_fig.update_layout(
        title_text='Effective Probability Mass Function',
        title_x=0.5,
        yaxis={'title': {'text': 'Probability'}}
    )

    return effective_logits_fig, effective_pmf_fig


def get_word_button_click(greedy_or_random_sampling_radio):
    global vocab, vocab_frequencies, original_logits, effective_logits, original_pmf, effective_pmf
    p = np.nan_to_num(effective_pmf, nan=0)
    if p.sum() == 0:
        raise gr.Error('Not a valid probability distribution!')
    else:
        if greedy_or_random_sampling_radio == 'Greedy Decoding':
            word = vocab[np.argmax(p)]
        elif greedy_or_random_sampling_radio == 'Random Sampling':
            word = np.random.choice(vocab, size=(1,), p=p).item()
    return word


gr.close_all()
with gr.Blocks() as app:
    gr.Markdown('# From Logits to Tokens')

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Column():
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
                repetition_penalty_checkbox = gr.Checkbox(
                    label='Repetition Penalty (RP)',
                    info='Check to enable, uncheck to disable',
                    value=True,
                    interactive=True
                )
                repetition_penalty_slider = gr.Slider(
                    minimum=0.01,
                    maximum=5,
                    step=0.01,
                    value=1,
                    interactive=True,
                    show_label=False
                )

            with gr.Column():
                frequency_penalty_checkbox = gr.Checkbox(
                    label='Frequency Penalty (FP)',
                    info='Check to enable, uncheck to disable',
                    value=True,
                    interactive=True
                )
                frequency_penalty_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=0.01,
                    value=0,
                    interactive=True,
                    show_label=False
                )

            with gr.Column():
                temperature_checkbox = gr.Checkbox(
                    label='Temperature (T)',
                    info='Check to enable, uncheck to disable',
                    value=True,
                    interactive=True
                )
                temperature_slider = gr.Slider(
                    minimum=0.01,
                    maximum=25,
                    step=0.01,
                    value=1,
                    interactive=True,
                    show_label=False
                )

            with gr.Column():
                top_k_or_top_p_radio = gr.Radio(
                    ['Top K', 'Top P (Nucleus)'],
                    info='Select to enable',
                    value='Top K',
                    interactive=True,
                    show_label=False
                )
                top_k_slider = gr.Slider(
                    label='Top K (top_k)',
                    minimum=1,
                    maximum=len(vocab),
                    step=1,
                    value=len(vocab),
                    interactive=True,
                    show_label=True
                )
                top_p_slider = gr.Slider(
                    label='Top P (top_p)',
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=1,
                    interactive=True,
                    show_label=True
                )

        with gr.Column(scale=20):
            original_logits_bar_plot = gr.Plot()
            effective_logits_bar_plot = gr.Plot()

        with gr.Column(scale=20):
            original_pmf_bar_plot = gr.Plot()
            effective_pmf_bar_plot = gr.Plot()

    with gr.Row():
        with gr.Column():
            greedy_or_random_sampling_radio = gr.Radio(
                ['Greedy Decoding', 'Random Sampling'],
                label='Select Word Selection Method',
                info='Select to enable',
                value='Greedy Decoding',
                interactive=True,
                show_label=True
            )
            get_word_button = gr.Button(
                value='Get Word',
                interactive=True
            )
        selected_word_textbox = gr.Textbox(
            label='Selected Word',
            interactive=False,
            show_label=True
        )

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

    greedy_or_random_sampling_radio.change(
        fn=get_word_button_click,
        inputs=[greedy_or_random_sampling_radio],
        outputs=[selected_word_textbox]
    )

    get_word_button.click(
        fn=get_word_button_click,
        inputs=[greedy_or_random_sampling_radio],
        outputs=[selected_word_textbox]
    )

app.launch(share=False)
