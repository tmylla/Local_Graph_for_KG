import numpy as np
from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, mode_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, opts=dict(
                legend=[mode_name],  # 每条曲线指定名字
                title=title_name,
                xlabel='Batchs',  # 横坐标名称
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=mode_name, update='append')
