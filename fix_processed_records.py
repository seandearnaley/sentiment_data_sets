def fix_processed_records(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    print("Original lines:", lines)  # Debugging: print original lines

    # Flatten the list and remove any array structures
    fixed_lines = []
    for line in lines:
        if line.startswith("[") and line.endswith(
            "]\n"
        ):  # Ensure it checks for newline at the end
            # Remove the brackets and split by comma
            line = line[1:-2]  # Adjust slicing to remove the newline as well
            ids = line.split('", "')
            fixed_lines.extend(ids)
        else:
            fixed_lines.append(line.strip())

    print("Fixed lines:", fixed_lines)  # Debugging: print fixed lines

    # Write the fixed lines back to the file
    with open(file_path, "w") as file:
        for id in fixed_lines:
            file.write(f"{id}\n")


# Usage
fix_processed_records("data/processed_records.txt")
