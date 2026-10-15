"""Image classification with transfer learning (MobileNetV2, flowers dataset).
Needs: pip install tensorflow. To do cats vs dogs instead, point `path`
at a folder with one sub-folder per class (e.g. cats/ and dogs/).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
path = tf.keras.utils.get_file("flower_photos", origin=URL, untar=True)
IMG, BS = (160, 160), 32
kw = dict(validation_split=0.2, seed=42, image_size=IMG, batch_size=BS)
train = tf.keras.utils.image_dataset_from_directory(path, subset="training", **kw)
val = tf.keras.utils.image_dataset_from_directory(path, subset="validation", **kw)
classes = train.class_names; print("Classes:", classes)
train, val = train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)

base = tf.keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False
aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1)])
inp = tf.keras.Input(IMG + (3,))
x = aug(inp)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
model = tf.keras.Model(inp, out)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
h = model.fit(train, validation_data=val, epochs=3)

plt.plot(h.history["accuracy"], label="train"); plt.plot(h.history["val_accuracy"], label="val")
plt.legend(); plt.title("MobileNetV2 transfer learning"); plt.savefig("results.png")
print("Final val accuracy: %.3f" % h.history["val_accuracy"][-1])
