"""script code to generate test data"""
import math

if __name__ == "__main__":
    for i in range(15):
        sq_i = math.sqrt(i)
        print(f"vtx{i:02d}, {(i % 2) == 0}, {(i % 3) == 0}, {i * i}, {sq_i}")