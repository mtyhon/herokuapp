import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import utils.frequencies as frequencies
import utils.mixed_modes_utils as mixed_modes_utils
import os

from plotly.colors import sample_colorscale
from astropy.timeseries import LombScargle
from copy import deepcopy
from os import path

def DeltaPi1_from_DeltaNu_RGB(DeltaNu):
    # Compute Period spacing (in s) from deltanu
    return 60 + 1.7*DeltaNu

def Lor_model(pds, peak):
    return peak.height / (1 + ((pds.frequency.values - peak.frequency)/peak.linewidth)**2)

def sinc2_model(pds, peak):
    deltanu = np.mean(np.diff(pds.frequency.values))
    return peak.height * np.sinc((pds.frequency.values - peak.frequency)/deltanu)**2

def fit_model(pds, peaks):

    model = np.ones_like(pds.frequency.values)

    for i in range(len(peaks)):
        if np.isfinite(peaks.linewidth.iloc[i]):
            model += Lor_model(pds, peaks.iloc[i,])
        else:
            model += sinc2_model(pds, peaks.iloc[i, ])
    return model
    

kicx = 3749487

#### TACO-Mosser ####

summary = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/raw/00%d/summary.csv' %kicx))
pds = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/raw/00%d/pds_bgr.csv' %kicx))
peaks = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/raw//00%d/peaksMLE.csv' %kicx))

# Only keep pds around oscillations
pds = pds.loc[abs(pds['frequency'].values - summary['numax'].values) < 3 * summary['sigmaEnv'].values, ]

peaks = peaks.loc[abs(peaks.frequency.values - summary.numax.values) < 3*summary.sigmaEnv.values, ]
l023_peaks = peaks.loc[(peaks.l == 0) | (peaks.l == 2) | (peaks.l == 3)]
l0_peaks = peaks.loc[(peaks.l==0), ]
l1_peaks = peaks.loc[(peaks.l == 1)  | (np.isfinite(peaks.l) == False)]
l2_peaks = peaks.loc[(peaks.l==2), ]



pds_l023_removed = pds.assign(power = pds.power / fit_model(pds, l023_peaks))
# Create artificial frequencies for creation of stretched power spectrum using values determined from TACO for this star
freqs = frequencies.Frequencies(frequency=pds_l023_removed.frequency.values,
                                numax=summary.numax.values, 
                                delta_nu=summary.DeltaNu.values if np.isfinite(summary.DeltaNu.values) else None, 
                                epsilon_p=summary.eps_p.values if np.isfinite(summary.eps_p.values) else None,
                                alpha=summary.alpha.values if np.isfinite(summary.alpha.values) else None)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title ='PSxPS'
server = app.server

app.layout = html.Div([
    
    html.Div([dcc.RadioItems(
                id='sample_type',
                options=[{'label': i, 'value': i} for i in ['BayesOpt', 'Sobol']],
                value='BayesOpt',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '95%', 'display': 'inline-block',
               'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
              }),
    
    
    html.Div([
        dcc.Graph(#figure=fig,
            id='turbo_samples',
            hoverData={'points': [{'x': df_comb.DPi1.values[np.argsort(df_comb.Loss.values)[len(df_comb)//2]],
                                  'y': df_comb.q.values[np.argsort(df_comb.Loss.values)[len(df_comb)//2]] }]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    
    html.Div([
        dcc.Graph(id='global_psps'),
        dcc.Graph(id='local_psps')
    ], style={'display': 'inline-block', 'width': '49%'}),
    

])



def create_psps(dpi, q, local=False):
    
    up_bound, low_bound = 200, 20 # RGB
    # up_bound, low_bound = 400, 100 # HeB

    params = {'calc_l0': True, 
                'calc_l2': True,
                'calc_l3': False, 
                'calc_nom_l1': True, 
                'calc_mixed': False,
                'calc_rot': False, 
                'DPi1': dpi,
                'coupling': q,
                'eps_g': 0.0, 
                'l': 1,
                }
    freqs(params)

    if pds_l023_removed.frequency.min() < freqs.l0_freqs.min():
        zeta = freqs.zeta[pds_l023_removed.frequency >= freqs.l0_freqs.min()]
        power = pds_l023_removed.power[pds_l023_removed.frequency >= freqs.l0_freqs.min()].values
        freq = pds_l023_removed.frequency[pds_l023_removed.frequency >= freqs.l0_freqs.min()].values
    else:
        power = pds_l023_removed.power.values
        freq = pds_l023_removed.frequency.values
        zeta = freqs.zeta

    new_frequency, tau, zeta = mixed_modes_utils.stretched_pds(freq, 
                                                               zeta)

    fr = np.arange(1/(up_bound), 1/(low_bound), 0.1/tau.max()) 
    ls = LombScargle(tau, power)
    PSD_LS = ls.power(fr)
    return 1/fr, PSD_LS
    
def format_psps(period, PSD_LS, dpi, q, click_period, click_power, tracecolor, click_tracecolor, local=False):
    
    if not local:
        xrange= [20, 150]
        titletext="Global"
        topm = 40
        xtitle=0.87
        xannot = 0.74
    else:
        xrange= [min(df_comb.DPi1), max(df_comb.DPi1)]
        titletext="Local"
        topm = 40  
        xtitle=0.853
        xannot= 0.732

    if len(click_period) == 0:
        data_ = [dict(
            x=period,
            y=PSD_LS,
            mode='line',
            line={'color': tracecolor,
                 'width': 2,
                 'coloraxis': 'coloraxis'},
            hovertemplate =
            '<b>DPi1</b>: %{x:.1f}s'+
            '<br><b>Power</b>: %{y:.4f}<extra></extra>', #<extra></extra> removes the 'Trace 0' in the hover
        )]
       
    else:
        data_ = [dict(
            x=period,
            y=PSD_LS,
            mode='line',
            line={'color': tracecolor,
                 'width': 2,
                 'coloraxis': 'coloraxis'},
            name='Current',
            hovertemplate =
            '<b>DPi1</b>: %{x:.1f}s'+
            '<br><b>Power</b>: %{y:.4f}<extra></extra>',
            showlegend = False

        ),
                dict(
            x=click_period,
            y=click_power,
            mode='line',
            line={'color': click_tracecolor,
                 'width': 2,
                 'coloraxis': 'coloraxis',
                 'dash': 'dash'},
            name='Focus',
            hovertemplate =
            '<b>DPi1</b>: %{x:.1f}s'+
            '<br><b>Power</b>: %{y:.4f}<extra></extra>',
            showlegend = False
        )]
        

    return {
        'data': data_,
        
        'layout': {
            'annotations': [{'x': 0.65, 'y': 0.75, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': ' PSxPS %s<br> Period Spacing: %.2fs<br> Coupling Factor: %.3f' %(titletext, dpi, q)
            }],

            'yaxis': {'type': 'linear',
                     "title": {'text': "Power", 'standoff': 1},
                     "range": [0, 0.006]},
            'xaxis': {'showgrid': False, 
                     "title": {'text': "Period Spacing (s)", 'standoff': 1},
                     "range": xrange},
            'hoverlabel': {'bgcolor': 'gray',
                           'namelength': 1,
                          'font': {'color': 'white'}},
            'margin' : {'t': topm, 'b': 40. ,'r': 10, 'l': 50},
            "height": 300,
            'shapes': [{'line': {'color': 'black', 'dash': 'longdashdot', 'width': 1.5},
                 'type': 'line',
                 'x0': dpi,
                 'x1': dpi,
                 'xref': 'x',
                 'y0': 0,
                 'y1': 0.006,
                 'yref': 'y'}],
                     
        }

    }


def format_scatter(df):    
    fig = px.scatter(df, x ='DPi1', y='q', title='Samples',
                color='Loss')
    fig.update_layout(
        yaxis_title='Coupling Factor q',
        xaxis_title='Period Spacing (s)',
        title_x=0.2,
        title_y=0.99,
        title_font_color = 'white',
        plot_bgcolor='ivory',
        height= 600,
        margin={'t': 20, 'l': 40, 'b': 5, 'r': 10}
    )

    fig.update_traces(hovertemplate =
        '<b>DPi1</b>: %{x:.1f}s'+
        '<br><b>q</b>: %{y:.3f}<br>')

    fig.update_xaxes(title_standoff=1, range=[min(df.DPi1), max(df.DPi1)], gridcolor='gainsboro') # Standoff is padding
    fig.update_yaxes(title_standoff=1, range=[min(df.q), max(df.q)], gridcolor='gainsboro') # Standoff is padding
    fig.update_coloraxes(showscale=False)

    return fig


@app.callback(
    dash.dependencies.Output('turbo_samples', 'figure'),
    [dash.dependencies.Input('sample_type', 'value')])
def update_scatter(samplename):
    inp_df = df_comb[df_comb.sample_type == samplename]
    return format_scatter(inp_df) 


@app.callback(
    [dash.dependencies.Output('global_psps', 'figure'), dash.dependencies.Output('local_psps', 'figure')],
    [dash.dependencies.Input('turbo_samples', 'clickData'),
     dash.dependencies.Input('turbo_samples', 'figure'), dash.dependencies.Input('local_psps', 'figure'),
    dash.dependencies.Input('global_psps', 'figure'), dash.dependencies.Input('global_psps', 'clickData')])
def update_psxps(clickData, inp_fig, local_fig, global_fig, global_clickData):

    try:
        dpi_value = clickData['points'][0]['x']
        q_value = clickData['points'][0]['y']
    except:
        soboldf = df_comb[df_comb.sample_type == 'Sobol']
        dpi_value =  soboldf.DPi1.values[np.argsort(soboldf.Loss.values)[len(soboldf)//2]]
        q_value = soboldf.q.values[np.argsort(soboldf.Loss.values)[len(soboldf)//2]]

    colorlist = inp_fig['data'][0]['marker']['color'] # loss values
    period, PSD_LS = create_psps(dpi_value, q_value)

    try:      
        markercol =  clickData['points'][0]['marker.color']
        _c = (markercol - np.min(colorlist)) / (np.max(colorlist) - np.min(colorlist))
        tracecolor = sample_colorscale(inp_fig['layout']['coloraxis']['colorscale'],
                                       [_c], low=0.0, high=1.0, colortype='rgb')[0]

    except:
        _c = (np.median(colorlist) - np.min(colorlist)) / (np.max(colorlist) - np.min(colorlist))
        tracecolor = sample_colorscale(inp_fig['layout']['coloraxis']['colorscale'],
                                       [_c], low=0.0, high=1.0, colortype='rgb')[0]
        
        
    if local_fig is not None:
        old_dpi, old_q = float(local_fig['layout']['annotations'][0]['text'].split(' ')[5][:5]),\
        float(local_fig['layout']['annotations'][0]['text'].split(' ')[8])
        click_period, click_power = create_psps(old_dpi, old_q)
        click_tracecolor = local_fig['data'][0]['line']['color']
        
    else:
        click_period, click_power, click_tracecolor = [], [], None
    
    
    forz1 = format_psps(period, PSD_LS,dpi_value, q_value, click_period, click_power,
                       tracecolor,click_tracecolor)
    forz2 = format_psps(period, PSD_LS, dpi_value, q_value,
                                                click_period, click_power, tracecolor, click_tracecolor, local=True)
    
            
    return forz1, forz2


if __name__ == '__main__':
    app.run_server()
