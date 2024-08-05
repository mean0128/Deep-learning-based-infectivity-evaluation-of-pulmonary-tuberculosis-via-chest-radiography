import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
import numpy as np
import random
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import Model

def grad_cam_plus_plus(model, img, layer_name, label_index):
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(img, axis=0))
        loss = predictions[:, label_index]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    
    max_val = np.percentile(heatmap, 95)
    heatmap = np.maximum(heatmap, max_val)
    return heatmap

#%%
random_state = 1
random.seed(random_state)

model = tf.keras.models.load_model("D:/wyj/DenseNet121_Final_Softmax.h5")

#%%
img = cv2.imread("D:/wyj/3798.png")
img = cv2.resize(img, (224,224))/255.
res_prob = model.predict(tf.expand_dims(img, 0))
analysingImg = img
res = tf.argmax(res_prob, axis=1).numpy()[0]
#%% LIME explaination
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(analysingImg, model.predict, top_labels=1, hide_color=0, num_samples=100, batch_size=100)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=1, hide_rest=True, positive_only=True)
img_boundary1 = mark_boundaries(temp, mask)

mask_Red = np.concatenate([np.expand_dims(mask, 2), np.zeros(np.expand_dims(mask, 2).shape), np.zeros(np.expand_dims(mask, 2).shape)], axis=2)
superimposed_img = mask_Red*0.15 + analysingImg 

#%% Grad-CAM++
heatmap = grad_cam_plus_plus(model, img, 'conv5_block16_concat', res)

#%% Grad-CAM++ and LIME Interpretation Visualzation
plt.figure(figsize=(6,6))
plt.imshow(superimposed_img)
plt.imshow(heatmap, cmap='jet', alpha=0.15)

plt.axis('off')
plt.title("Prediction result is %s." % ('positive' if res else 'negative'))
plt.show()