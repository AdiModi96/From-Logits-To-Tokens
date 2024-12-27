import os
import random

from functions import *
import project_paths as pp
import plotly.graph_objects as go
import gradio as gr

with open(os.path.join(pp.datasets_folder_path, 'words.txt')) as file:
    words = file.read().splitlines()

vocab_size = 50

random.seed(vocab_size)
vocab = sorted(random.sample(words, vocab_size))
vocab_frequencies = np.zeros(shape=vocab_size, dtype=np.int16)

np.random.seed(vocab_size)
original_logits = np.random.normal(0, 0.11, size=vocab_size)
effective_logits_pre_pruning = original_logits.copy()
effective_logits_post_pruning = original_logits.copy()
original_pmf = convert_logits_to_probabilities(original_logits)
effective_pmf_pre_pruning = original_pmf.copy()
effective_pmf_post_pruning = original_pmf.copy()


def update_original_plots():
    global vocab, vocab_frequencies
    global original_logits, original_pmf

    original_logits_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=original_logits,
            marker={
                'color': original_logits,
                'colorscale': 'YlOrBr'
            },
            marker_line={'width': 0.25, 'color': '#666666'}
        )
    )
    original_logits_fig.update_layout(
        title_text='Original Logits Distribution',
        title_x=0.5,
        yaxis={'title': {'text': 'Logits'}},
        plot_bgcolor='#DBDBDB'
    )

    original_pmf_fig = go.Figure(
        go.Bar(
            x=vocab,
            y=original_pmf,
            marker={
                'color': original_pmf,
                'colorscale': 'YlOrBr'
            },
            marker_line={'width': 0.25, 'color': '#666666'}
        )
    )
    original_pmf_fig.update_layout(
        title_text='Original Probability Mass Function',
        title_x=0.5,
        yaxis={'title': {'text': 'Probability'}},
        plot_bgcolor='#DBDBDB'
    )

    return original_logits_fig, original_pmf_fig


def vocab_dropdown_change(vocab_dropdown):
    global vocab, vocab_frequencies
    return vocab_frequencies[vocab.index(vocab_dropdown)]


def vocab_frequency_slider_release(
        vocab_dropdown,
        vocab_frequency_slider,
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        distribution_pruning_strategy_dropdown,
        distribution_pruning_strategy_slider
):
    global vocab, vocab_frequencies
    vocab_frequencies[vocab.index(vocab_dropdown)] = vocab_frequency_slider
    return update_effective_plots(
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        distribution_pruning_strategy_dropdown,
        distribution_pruning_strategy_slider
    )


def update_distributions(
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        distribution_pruning_strategy_dropdown,
        distribution_pruning_strategy_slider
):
    global vocab, vocab_frequencies
    global original_logits, effective_logits_pre_pruning, effective_logits_post_pruning
    global original_pmf, effective_pmf_pre_pruning, effective_pmf_post_pruning

    effective_logits_pre_pruning = original_logits.copy()
    if repetition_penalty_checkbox:
        effective_logits_pre_pruning = apply_repetition_penalty(effective_logits_pre_pruning, vocab_frequencies, repetition_penalty_slider)
    if frequency_penalty_checkbox:
        effective_logits_pre_pruning = apply_frequency_penalty(effective_logits_pre_pruning, vocab_frequencies, frequency_penalty_slider)
    if temperature_checkbox:
        effective_logits_pre_pruning = apply_temperature(effective_logits_pre_pruning, temperature_slider)
    effective_pmf_pre_pruning = convert_logits_to_probabilities(effective_logits_pre_pruning)

    if distribution_pruning_strategy_dropdown == 'Top K':
        effective_logits_post_pruning = select_top_k(effective_logits_pre_pruning, distribution_pruning_strategy_slider)
    elif distribution_pruning_strategy_dropdown == 'Top P':
        effective_logits_post_pruning = select_top_p(effective_logits_pre_pruning, distribution_pruning_strategy_slider)
    effective_pmf_post_pruning = convert_logits_to_probabilities(effective_logits_post_pruning)


def update_effective_plots(
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        distribution_pruning_strategy_dropdown,
        distribution_pruning_strategy_slider
):
    update_distributions(
        repetition_penalty_checkbox,
        repetition_penalty_slider,
        frequency_penalty_checkbox,
        frequency_penalty_slider,
        temperature_checkbox,
        temperature_slider,
        distribution_pruning_strategy_dropdown,
        distribution_pruning_strategy_slider
    )

    def update_pre_pruning_plots():
        effective_logits_pre_pruning_fig = go.Figure(
            go.Bar(
                x=vocab,
                y=effective_logits_pre_pruning,
                marker={
                    'color': effective_logits_pre_pruning,
                    'colorscale': 'YlOrBr'
                },
                marker_line={'width': 0.25, 'color': '#666666'}
            )
        )
        effective_logits_pre_pruning_fig.update_layout(
            autosize=True,
            title_text='Effective Logits Distribution',
            title_x=0.5,
            yaxis={'title': {'text': 'Logits'}},
            plot_bgcolor='#DBDBDB'
        )

        effective_pmf_pre_pruning_fig = go.Figure(
            go.Bar(
                x=vocab,
                y=effective_pmf_pre_pruning,
                marker={
                    'color': effective_pmf_pre_pruning,
                    'colorscale': 'YlOrBr'
                },
                marker_line={'width': 0.25, 'color': '#666666'}
            )
        )
        effective_pmf_pre_pruning_fig.update_layout(
            autosize=True,
            title_text='Effective Probability Mass Function',
            title_x=0.5,
            yaxis={'title': {'text': 'Probability'}},
            plot_bgcolor='#DBDBDB'
        )

        return [effective_logits_pre_pruning_fig, effective_pmf_pre_pruning_fig]

    def update_distribution_pruning_statistics_plot():
        if distribution_pruning_strategy_dropdown == 'Top K':
            sorted_idxes = np.argsort(effective_logits_pre_pruning)[::-1]
            sorted_vocab = [vocab[i] for i in sorted_idxes]
            k = np.arange(start=1, stop=vocab_size + 1, step=1, dtype=np.uint16)
            distribution_pruning_statistics_plot = go.Figure(
                go.Scatter(
                    x=sorted_vocab,
                    y=k,
                    mode='markers+lines',
                    marker_color=[
                        ('#662506' if value <= distribution_pruning_strategy_slider else '#FFEEA9') for value in k
                    ],
                    line_color='#F97316',
                    marker_line={'width': 0.25, 'color': '#662506'}
                )
            )
            distribution_pruning_statistics_plot.update_layout(
                autosize=True,
                title_text='Top K Value Distribution',
                title_x=0.5,
                yaxis={'title': {'text': 'Top K'}},
                plot_bgcolor='#DBDBDB'
            )
        elif distribution_pruning_strategy_dropdown == 'Top P':
            sorted_idxes = np.argsort(effective_pmf_pre_pruning)[::-1]
            cumulative_pmf_pre_pruning = np.cumsum(effective_pmf_pre_pruning[sorted_idxes])
            sorted_vocab = [vocab[i] for i in sorted_idxes]
            distribution_pruning_statistics_plot = go.Figure(
                go.Scatter(
                    x=sorted_vocab,
                    y=cumulative_pmf_pre_pruning,
                    mode='markers+lines',
                    marker_color=[
                        ('#662506' if value <= distribution_pruning_strategy_slider else '#FFEEA9') for value in cumulative_pmf_pre_pruning
                    ],
                    line_color='#F97316',
                    marker_line={'width': 0.25, 'color': '#662506'}
                )
            )
            distribution_pruning_statistics_plot.update_layout(
                autosize=True,
                title_text='Cumulative PMF Distribution',
                title_x=0.5,
                yaxis={'title': {'text': 'Cumulative PMF'}},
                plot_bgcolor='#DBDBDB'
            )

        return [distribution_pruning_statistics_plot]

    def update_post_pruning_plots():
        effective_logits_post_pruning_fig = go.Figure(
            go.Bar(
                x=vocab,
                y=effective_logits_post_pruning,
                marker={
                    'color': effective_logits_post_pruning,
                    'colorscale': 'YlOrBr'
                },
                marker_line={'width': 0.25, 'color': '#666666'}
            )
        )
        effective_logits_post_pruning_fig.update_layout(
            autosize=True,
            title_text='Effective Logits Distribution',
            title_x=0.5,
            yaxis={'title': {'text': 'Logits'}},
            plot_bgcolor='#DBDBDB'
        )

        effective_pmf_post_pruning_fig = go.Figure(
            go.Bar(
                x=vocab,
                y=effective_pmf_post_pruning,
                marker={
                    'color': effective_pmf_post_pruning,
                    'colorscale': 'YlOrBr'
                },
                marker_line={'width': 0.25, 'color': '#666666'}
            )
        )
        effective_pmf_post_pruning_fig.update_layout(
            autosize=True,
            title_text='Effective Probability Mass Function',
            title_x=0.5,
            yaxis={'title': {'text': 'Probability'}},
            plot_bgcolor='#DBDBDB'
        )

        return [effective_logits_post_pruning_fig, effective_pmf_post_pruning_fig]

    return update_pre_pruning_plots() + update_distribution_pruning_statistics_plot() + update_post_pruning_plots()


def distribution_pruning_strategy_dropdown_change(distribution_pruning_strategy_dropdown):
    if distribution_pruning_strategy_dropdown == 'Top K':
        updated_distribution_pruning_strategy_slider = gr.update(
            minimum=1,
            maximum=len(vocab),
            step=1,
            value=len(vocab)
        )
    elif distribution_pruning_strategy_dropdown == 'Top P':
        updated_distribution_pruning_strategy_slider = gr.update(
            minimum=0,
            maximum=1,
            step=0.01,
            value=1
        )

    return updated_distribution_pruning_strategy_slider


def get_word_button_click(greedy_or_random_sampling_radio):
    global vocab, vocab_frequencies
    global original_logits, effective_logits_pre_pruning, effective_logits_post_pruning
    global original_pmf, effective_pmf_pre_pruning, effective_pmf_post_pruning

    p = np.nan_to_num(effective_pmf_post_pruning, nan=0)
    if p.sum() == 0:
        raise gr.Error('Not a valid probability distribution!')
    else:
        if greedy_or_random_sampling_radio == 'Greedy Decoding':
            word = vocab[np.argmax(p)]
        elif greedy_or_random_sampling_radio == 'Random Sampling':
            word = np.random.choice(vocab, size=(1,), p=p).item()
    return word


gr.close_all()
with (gr.Blocks(fill_width=True, css='footer {visibility: hidden}') as app):
    gr.Markdown('# From Logits to Tokens')
    with gr.Accordion(
        label='Details',
        open=False
    ):
        gr.Markdown(
            'A simple app that illustrates the process of converting raw logits from an LLM\'s decoder layer to the final token.\\\n' +
            'For simplicity and ease of understanding, we\'ll assume tokens to be complete words with a vocabulary size of 50.\\\n' +
            'If you are interested in learning the following in detail, you can ready my blog post: https://medium.com/@adimodi96/from-logits-to-tokens-9a36feab9cab.'
        )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column():
                vocab_dropdown = gr.Dropdown(
                    label='Token/Word',
                    info='Select a token/word to set its frequency, i.e., the number of time token/word appeared before in the sequence.',
                    choices=vocab,
                    value=vocab[0],
                    interactive=True
                )

                vocab_frequency_slider = gr.Slider(
                    label='Token/Word Frequency',
                    info='Set the frequency of the above selected token/word',
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
                distribution_pruning_strategy_dropdown = gr.Dropdown(
                    choices=['Top K', 'Top P'],
                    label='Distribution Pruning',
                    info='Select a distribution pruning strategy',
                    value='Top K',
                    interactive=True,
                    show_label=True
                )
                distribution_pruning_strategy_slider = gr.Slider(
                    minimum=1,
                    maximum=len(vocab),
                    step=1,
                    value=len(vocab),
                    interactive=True,
                    show_label=False
                )

        with gr.Column(scale=3):
            with gr.Row():
                original_logits_bar_plot = gr.Plot()
                original_pmf_bar_plot = gr.Plot()

            with gr.Row():
                with gr.Tab('Effective Distribution Pre-Pruning') as pre_pruning_plots_tab:
                    with gr.Row():
                        effective_logits_bar_plot_pre_pruning = gr.Plot()
                        effective_pmf_bar_plot_pre_pruning = gr.Plot()
                with gr.Tab('Pruning Statistics') as pruning_statistics_plot_tab:
                    with gr.Row():
                        distribution_pruning_statistics_plot = gr.Plot()
                with gr.Tab('Effective Distribution Post-Pruning') as post_pruning_plots_tab:
                    with gr.Row():
                        effective_logits_bar_plot_post_pruning = gr.Plot()
                        effective_pmf_bar_plot_post_pruning = gr.Plot()

    with gr.Row():
        with gr.Column():
            greedy_or_random_sampling_radio = gr.Radio(
                ['Greedy Decoding', 'Random Sampling'],
                label='Select Token/Word Selection Method',
                value='Greedy Decoding',
                interactive=True,
                show_label=True
            )
            get_word_button = gr.Button(
                value='Get Token/Word',
                interactive=True
            )
        with gr.Column():
            selected_word_textbox = gr.Textbox(
                label='Selected Token/Word',
                interactive=False,
                show_label=True
            )
            gr.Markdown(
                value='Built by **Aditya Modi** ' +
                      '• [E-Mail](mailto:aditya.modi.in@example.com) ' +
                      '• [GitHub](https://github.com/AdiModi96/From-Logits-To-Tokens) ' +
                      '• [LinkedIn](https://www.linkedin.com/in/aditya-modi-in)',
                show_label=False
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
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )

    vocab_dropdown.change(
        fn=vocab_dropdown_change,
        inputs=[vocab_dropdown],
        outputs=[vocab_frequency_slider]
    )
    vocab_frequency_slider.release(
        fn=vocab_frequency_slider_release,
        inputs=[
            vocab_dropdown,
            vocab_frequency_slider,
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
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
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )
    repetition_penalty_slider.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
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
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )
    frequency_penalty_slider.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
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
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )
    temperature_slider.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )

    distribution_pruning_strategy_dropdown.change(
        fn=distribution_pruning_strategy_dropdown_change,
        inputs=[distribution_pruning_strategy_dropdown],
        outputs=[distribution_pruning_strategy_slider]
    )
    distribution_pruning_strategy_slider.change(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )

    pre_pruning_plots_tab.select(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )
    pruning_statistics_plot_tab.select(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
    )
    post_pruning_plots_tab.select(
        fn=update_effective_plots,
        inputs=[
            repetition_penalty_checkbox,
            repetition_penalty_slider,
            frequency_penalty_checkbox,
            frequency_penalty_slider,
            temperature_checkbox,
            temperature_slider,
            distribution_pruning_strategy_dropdown,
            distribution_pruning_strategy_slider
        ],
        outputs=[
            effective_logits_bar_plot_pre_pruning,
            effective_pmf_bar_plot_pre_pruning,
            distribution_pruning_statistics_plot,
            effective_logits_bar_plot_post_pruning,
            effective_pmf_bar_plot_post_pruning
        ]
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

app.launch(share=True)
