import sys
import traceback

with open("out.txt", "w") as f:
    sys.stdout = f
    sys.stderr = f
    try:
        sys.path.append('src')
        from train_model import train_and_evaluate
        print("Successfully imported train_and_evaluate")
        train_and_evaluate()
        print("Training finished.")
    except Exception as e:
        print("Error occurred:")
        traceback.print_exc()
