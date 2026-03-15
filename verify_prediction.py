import sys
with open("verify_out.txt", "w") as f:
    try:
        sys.path.append('src')
        from predictor import CommentPredictor
        p = CommentPredictor()
        result = p.predict("This is a great tutorial, very informative!")
        f.write(f"Prediction result: {result}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
