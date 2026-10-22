from api.predictor import ImageClassifier  # adjust to wherever you saved the file

clf = ImageClassifier()
with open("data/Train/Image_1.jpg", "rb") as f:
    print(clf.predict_from_file(f))