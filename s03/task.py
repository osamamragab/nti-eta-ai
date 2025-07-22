import pandas as pd


def main():
    filename = input("file name: ")
    if not filename.endswith(".xlsx"):
        filename += ".xlsx"

    data = []
    while True:
        try:
            line = input(f"{len(data) + 1}: ")
            if line.strip().lower() in ["quit", "exit"]:
                break
            row = {}
            for i, field in enumerate(line.split(",")):
                row[f"field {i + 1}"] = field.strip()
            data.append(row)
        except EOFError:
            print("")
            break
        except KeyboardInterrupt:
            print("\ninterrupted")
            break

    if not data:
        print("you kidding me?")
        return

    df = pd.DataFrame(data)
    try:
        df.to_excel(filename, index=False, engine="openpyxl")
        print(f"wrote {len(data)} lines to file '{filename}'")
    except Exception as e:
        print(f"failed to save exel file: {e}")


if __name__ == "__main__":
    main()
