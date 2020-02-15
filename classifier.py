#!/usr/bin/env python
# coding: utf-8

# Object classification for self-driving cars

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[3]:


bs = 10


# In[4]:


path = '/home/riya/datasets/road_object_classification/train'


# In[5]:


get_ipython().system('ls $path')


# In[6]:


fnames = get_image_files(os.path.join(path,'animal'))
fnames[:5]


# In[7]:


data = ImageDataBunch.from_folder('/home/riya/datasets/road_object_classification/train', train=".", valid_pct=0.2,
        ds_tfms=get_transforms(),bs=bs,size=224,num_workers=4).normalize(imagenet_stats)


# In[8]:


data.show_batch(rows=3, figsize=(7,6))


# In[9]:


print(data.classes)
len(data.classes),data.c


# In[10]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[11]:


learn.model


# In[12]:


learn.fit_one_cycle(6)


# In[ ]:


learn.save('stage-1')


# In[14]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[15]:


interp.plot_top_losses(9, figsize=(15,11))


# In[16]:


doc(interp.plot_top_losses)


# In[17]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[18]:


interp.most_confused(min_val=2)


# In[19]:


learn.unfreeze()


# In[20]:


learn.fit_one_cycle(1)


# That's a pretty accurate model!

# In[22]:


learn.lr_find()


# In[23]:


learn.unfreeze()


# In[24]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[25]:


learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-1))


# In[ ]:




