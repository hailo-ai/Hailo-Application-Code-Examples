import time
import json
import numpy as np
import os
from PIL import Image
import logging
from clip_app.logger_setup import setup_logger, set_log_level

# This class is used to store the text embeddings and match them to image embeddings
# The class can be initialized with an ONNX model or a CLIP model

# This class should be used as a singleton!
# An instance of this class is created in the end of this file.
# import text_image_matcher from this file to make sure that only one instance of the TextImageMatcher class is created.
# Example: from TextImageMatcher import text_image_matcher

# Set up the logger
logger = setup_logger()
# Change the log level to INFO
set_log_level(logger, logging.INFO)

# Set up global variables. Only required imports are done in the init functions
clip = None
ort = None
torch = None


class TextEmbeddingEntry:
    def __init__(self, text="", embedding=None, negative=False, ensemble=False):
        self.text = text
        self.embedding = embedding if embedding is not None else np.array([])
        self.negative = negative
        self.ensemble = ensemble
        self.probability = 0.0
    
    def to_dict(self):
        return {
            "text": self.text,
            "embedding": self.embedding.tolist(),  # Convert numpy array to list
            "negative": self.negative,
            "ensemble": self.ensemble
        }

class Match:
    def __init__(self, row_idx, text, similarity, entry_index, negative, passed_threshold):
        self.row_idx = row_idx # row index in the image embedding
        self.text = text # best matching text
        self.similarity = similarity # similarity between the image and best text embeddings
        self.entry_index = entry_index # index of the entry in TextImageMatcher.entries
        self.negative = negative # True if the best match is a negative entry
        self.passed_threshold = passed_threshold # True if the similarity is above the threshold

    def to_dict(self):
        return {
            "row_idx": self.row_idx,
            "text": self.text,
            "similarity": self.similarity,
            "entry_index": self.entry_index,
            "negative": self.negative,
            "passed_threshold": self.passed_threshold
        }

class TextImageMatcher:
    def __init__(self, model_name="RN50x4", threshold=0.8, max_entries=6):
        self.model = None # model is initialized in init_onnx or init_clip
        self.model_runtime = None
        self.model_name = model_name
        self.threshold = threshold
        self.run_softmax = True
        self.entries = [TextEmbeddingEntry() for _ in range(max_entries)]
        self.user_data = None # user data can be used to store additional information (used by multistream to save current stream id)
        self.text_prefix = "A photo of a "
        self.ensemble_template = [
            'a photo of a {}.',
            'a photo of the {}.',
            'a photo of my {}.',
            'a photo of a big {}.',
            'a photo of a small {}.',
        ]
    # Define class as a singleton
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextImageMatcher, cls).__new__(cls)
        return cls._instance

    def init_onnx(self, onnx_model_path):
        global clip, ort
        import clip
        import onnxruntime as ort
        print(f"Loading ONNX model this might take a while...")
        self.model = ort.InferenceSession(onnx_model_path)
        self.model_runtime = "onnx"

    def init_clip(self):
        global clip, torch
        import clip
        import torch
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        print(f"Loading model {self.model_name} on device {device} this might take a while...")
        self.model, self.preprocess = clip.load(self.model_name, device=device)
        self.device = device
        self.model_runtime = "clip"

    def set_threshold(self, new_threshold):
        self.threshold = new_threshold
    
    def set_text_prefix(self, new_text_prefix):
        self.text_prefix = new_text_prefix
    
    def set_ensemble_template(self, new_ensemble_template):
        self.ensemble_template = new_ensemble_template

    def update_text_entries(self, new_entry, index=None):
        if index is None:
            for i, entry in enumerate(self.entries):
                if entry.text == "":
                    self.entries[i] = new_entry
                    return
            print("Error: Entry list is full.")
        elif 0 <= index < len(self.entries):
            self.entries[index] = new_entry
        else:
            print(f"Error: Index out of bounds: {index}")

    def add_text(self, text, index=None, negative=False, ensemble=False):
        global clip, torch
        if self.model_runtime is None:
            print("Error: No model is loaded. Please call init_onnx or init_clip before calling add_text.")
            return
        if ensemble:
            text_entries = [template.format(text) for template in self.ensemble_template]
        else:
            text_entries = [self.text_prefix + text]
        logger.debug(f"Adding text entries: {text_entries}")
        if self.model_runtime == "onnx":
            text_tokens = clip.tokenize(text_entries)
            # Convert to numpy int64 array, as expected by the model
            text_tokens = text_tokens.numpy().astype(np.int64)
            text_features = self.model.run(None, {'input': text_tokens})[0]
            norm = np.linalg.norm(text_features, axis=-1, keepdims=True)
            text_features /= norm
            ensemble_embedding = np.mean(text_features, axis=0).flatten()
        else:
            text_tokens = clip.tokenize(text_entries).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                ensemble_embedding = torch.mean(text_features, dim=0).cpu().numpy().flatten()
        new_entry = TextEmbeddingEntry(text, ensemble_embedding, negative, ensemble)
        self.update_text_entries(new_entry, index)

    def get_embeddings(self):
        # return a list of indexes to self.entries if entry.text != ""
        valid_entries = [i for i, entry in enumerate(self.entries) if entry.text != ""]

        return valid_entries

    def get_texts(self):
        # returns all entries text (not only valid ones)
        return [entry.text for entry in self.entries]
    
    def save_embeddings(self, filename):
        # Prepare a dictionary that includes all the required data
        data_to_save = {
            "threshold": self.threshold,
            "text_prefix": self.text_prefix,
            "ensemble_template": self.ensemble_template,
            "entries": [entry.to_dict() for entry in self.entries]
        }

        # Save the dictionary as JSON
        with open(filename, 'w') as f:
            json.dump(data_to_save, f)

    def load_embeddings(self, filename):
        # if file does not exist create it
        if not os.path.isfile(filename):
            # File does not exist, create it
            with open(filename, 'w') as file:
                file.write('')  # Create an empty file or initialize with some data
            print(f"File {filename} does not exist, creating it.")
        else:
            try:
                # File exists, load the data
                with open(filename, 'r') as f:
                    data = json.load(f)

                    self.threshold = data['threshold']
                    self.text_prefix = data['text_prefix']
                    self.ensemble_template = data['ensemble_template']
                    
                    # Assuming TextEmbeddingEntry is a class that can be initialized like this
                    self.entries = [TextEmbeddingEntry(text=entry['text'], 
                                                    embedding=np.array(entry['embedding']), 
                                                    negative=entry['negative'],
                                                    ensemble=entry['ensemble']) 
                                    for entry in data['entries']]
            except Exception as e:
                print(f"Error while loading file {filename}: {e}. Maybe you forgot to save your embeddings?")

    def get_image_embedding(self, image):
        if self.model_runtime is None:
            print("Error: No model is loaded. Please call init_clip before calling get_image_embedding.")
            return
        if self.model_runtime == "onnx":
            print("Error: get_image_embedding is not supported for ONNX models.")
            return
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding.cpu().numpy().flatten()     

    def match(self, image_embedding_np, report_all=False):
        # This function is used to match an image embedding to a text embedding
        # Returns a list of tuples: (row_idx, text, similarity, enrty_index)
        # row_idx is the index of the row in the image embedding
        # text is the best matching text
        # similarity is the similarity between the image and text embeddings
        # entry_index is the index of the entry in self.entries
        # If the best match is a negative entry, or if the similarity is below the threshold, the tuple is not returned
        # If no match is found, an empty list is returned
        # If report_all is True, the function returns a list of all matches,
        # including negative entries and entries below the threshold.

        if len(image_embedding_np.shape) == 1:
            image_embedding_np = image_embedding_np.reshape(1, -1)
        results = []
        all_dot_products = None
        # set_breakpoint_every_n_frames()
        valid_entries = self.get_embeddings()
        if len(valid_entries) == 0:
            return []
        text_embeddings_np = np.array([self.entries[i].embedding for i in valid_entries])
        for row_idx, image_embedding_1d in enumerate(image_embedding_np):
            dot_products = np.dot(text_embeddings_np, image_embedding_1d)
            # add dot_products to all_dot_products as new line
            if all_dot_products is None:
                all_dot_products = dot_products[np.newaxis, :]
            else:
                all_dot_products = np.vstack((all_dot_products, dot_products))
            
            if self.run_softmax:
                # Compute softmax for each row (i.e. each image embedding)
                similarities = np.exp(100 * dot_products)
                similarities /= np.sum(similarities)
            else:
                # stats min: 0.27013595659637846, max: 0.4043235050452188, avg: 0.33676838831786493
                # map to [0,1]
                similarities = (dot_products - 0.27) / (0.41 - 0.27)
                # clip to [0,1]
                similarities = np.clip(similarities, 0, 1)
        
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            for i, value in enumerate(similarities):
                self.entries[valid_entries[i]].probability = similarities[i]
            new_match = Match(row_idx, 
                            self.entries[valid_entries[best_idx]].text, 
                            best_similarity, valid_entries[best_idx], 
                            self.entries[valid_entries[best_idx]].negative, 
                            best_similarity > self.threshold)
            if not report_all and new_match.negative:
                # Background is the best match
                continue
            if report_all or new_match.passed_threshold:
                results.append(new_match)
        
        logger.debug(f"Best match output: {results}")
        return results
    
# Instantiate the TextImageMatcher class to make sure that only one instance of the TextImageMatcher class is created.
text_image_matcher = TextImageMatcher()

def main():
    # get cli args
    import argparse
    parser = argparse.ArgumentParser()
    # add onnx_path arg
    # Underdevelopment #
    # parser.add_argument("--onnx", action="store_true", help="use onnx model, requires onnx-path")
    # parser.add_argument("--onnx-path", type=str, default="textual.onnx", help="path to onnx model")
    # Underdevelopment #

    parser.add_argument("--output", type=str, default="text_embeddings.json", help="output file name default=text_embeddings.json")
    parser.add_argument("--interactive", action="store_true", help="input text from interactive shell")
    parser.add_argument("--image-path", type=str, default=None, help="Optional, path to image file to match. Note image embeddings are not running on Hailo here.")
    # add text-input-list arg which take a list of texts to add
    parser.add_argument('--texts-list', nargs='+', help='A list of texts to add to the matcher, the first one will be the searched text, the others will be considered negative prompts.\n Example: --texts-list "cat" "dog" "yellow car"')
    # get args
    args = parser.parse_args()

    # Initialize the matcher and add text embeddings
    matcher = TextImageMatcher()
    matcher.init_clip()
    texts = []
    if args.interactive:
        while True:
            text = input(f'Enter text (leave empty to finish) {matcher.text_prefix}: ')
            if text == "":
                break
            texts.append(text)
    else: 
        if args.texts_list is not None:
            texts = args.texts_list
        else:
            texts = [
                "birthday cake",
                "person",
                "landscape",
            ]

    print(f"Adding text embeddings: ")
    first = True
    for text in texts:
        if first:
            print(f'{matcher.text_prefix}{text} (positive)')
            first = False
        else:
            print(f'{matcher.text_prefix}{text} (negative)')
    # Measure the time taken for the add_text function
    start_time = time.time()
    first = True
    for text in texts:
        if first:
            matcher.add_text(text)
            first = False
        else:
            matcher.add_text(text, negative=True)
    end_time = time.time()
    print(f"Time taken to add {len(texts)} text embeddings using add_text(): {end_time - start_time:.4f} seconds")

    matcher.save_embeddings(args.output)

    if args.image_path is None:
        print("No image path provided, skipping image embedding generation")
        exit()
    # Read an image from file
    image = Image.open(args.image_path)

    # Generate image embedding using the new method
    image_embedding = matcher.get_image_embedding(image)

    # Measure the time taken for the match function
    start_time = time.time()
    result = matcher.match(image_embedding, report_all=True)
    end_time = time.time()
    # Output the results
    print(f"Best match: {result[0].text}")
    valid_entries = matcher.get_embeddings()
    for i in valid_entries:
        print(f"Entry {i}: {matcher.entries[i].text} similarity: {matcher.entries[i].probability:.4f}")
    print(f"Time taken to run match(): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()