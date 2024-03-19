# This is how we compute the empty embedding
# You may need to download 'openai/clip-vit-large-patch14'

from ldm.modules.encoders.modules import FrozenCLIPEmbedder

model = FrozenCLIPEmbedder().to("cuda")
embedding = model.encode([""]).cpu()

print(embedding)
print(embedding.shape)

import pickle
with open("empty_embedding.pkl", 'wb') as f:
    pickle.dump(embedding, f)