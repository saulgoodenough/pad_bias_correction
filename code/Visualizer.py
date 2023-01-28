import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    orginal visdom API - self.vis.function
    '''

    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        # save（’loss',23）
        self.index = {} 
        self.log_text = ''
    
    def check_connection(self):
        return self.vis.check_connection()

    def plot(self, name, value,xlabel='Epoch',ylabel='Training loss',step=1,**kwargs):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([value]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name,xlabel=xlabel,ylabel=ylabel),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + step
        
    def plot_all(self, dict, step=1):
        '''
        plot multiple graphs
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in dict.items():
            self.plot(k, v, step)
            
    def plot_combine(self, name, d, step=1):
        #multiple plots in one single graph
        x = self.index.get(name, 0)
        X = []
        Y = []
        legend = []
        for k, v in sorted(d.items()):
            Y.append(v)
            X.append(x)
            legend.append(k)
        Y = np.array([Y])
        X = np.array([X])
        self.vis.line(
            Y=Y, 
            X=X,
            win=name,
            opts=dict(
                title=name,
                legend=legend,
                xlabel='epoch',
                ylabel='batch loss'
            ),
            update=None if x == 0 else 'append',
        )
        self.index[name] = x + step

    def img(self, name, img_,**kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrow=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrow=10)~~！！！
        '''
        self.vis.images(img_.cpu().numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs
                       )
                       
    def img_all(self, d):
        for k, v in d.iteritems():
            self.img(k, v)


    def log(self,info,name='log'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(self.log_text,name)   

    def __getattr__(self, name):
        return getattr(self.vis, name)
