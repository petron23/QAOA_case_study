from typing import Union, List
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import plotly.graph_objects as go
from qiskit.circuit import QuantumCircuit


def plot_charging_schedule(charging_unit, minimization_result_x, marker_size=50,) -> go.Figure:
    marker_colors = ["green", "red", "orange", "blue", "magenta","goldenrod"]
    time_slots = np.arange(0, charging_unit.num_time_slots)
    fig = go.Figure()
    already_in_legend = []
    for t in time_slots:
        offset = 0
        for car_num in np.arange(0, len(charging_unit.cars_to_charge)):
            car_id_current_car = charging_unit.cars_to_charge[car_num].car_id
            minimization_result_x_current_car = minimization_result_x[car_num*charging_unit.num_time_slots:(car_num+1)*charging_unit.num_time_slots]
            power_t = minimization_result_x_current_car[t]
            fig.add_trace(go.Scatter(
                x=[t+0.5]*int(power_t),
                y=offset + np.arange(0, power_t),
                mode="markers",
                marker_symbol="square",
                marker_size=marker_size,
                marker_color=marker_colors[car_num],
                name=car_id_current_car,
                showlegend=False if car_id_current_car in already_in_legend 
                else True))
            offset += power_t
            if power_t > 0:
                already_in_legend.append(car_id_current_car)    
    fig.update_xaxes(
        tick0=1,
        dtick=1,
        range=[0.01, charging_unit.num_time_slots],
        tickvals=np.arange(0.5, charging_unit.num_time_slots),
        ticktext=np.arange(0, charging_unit.num_time_slots),
        title="time slot",
        title_font_size=12,
        )
    fig.update_yaxes(
        range=[-0.6, charging_unit.num_charging_levels-1],
        tickvals=np.arange(-0.5, charging_unit.num_charging_levels-0.5),
        ticktext=np.arange(0, charging_unit.num_charging_levels),
        title="charging level",
        title_font_size=12,
        zeroline=False
        )
    return fig
