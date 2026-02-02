import os


def write_model_h_file(path: str, defines: dict, declarations: list[str]):
    # Ensure that the folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write header file
    with open(path, "w") as h_file:
        h_file.write("#ifndef MODEL_H\n")
        h_file.write("#define MODEL_H\n")
        h_file.write("\n")
        for key, value in defines.items():
            h_file.write(f'#define {key} {value}\n')
        h_file.write("\n")
        for declaration in declarations:
            h_file.write(f'{declaration}\n')
        h_file.write("\n")
        h_file.write("extern const unsigned char model_binary[];\n")
        h_file.write("\n")
        h_file.write("#endif\n")


def write_model_c_file(path: str, tflite_model):
    # Ensure that the folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write source file
    with open(path, "w") as c_file:
        c_file.write("const unsigned char model_binary[] = {\n")
        for i, byte in enumerate(tflite_model):
            c_file.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                c_file.write("\n")
        c_file.write("\n};\n")