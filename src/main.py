from utils import summarize_pdf_on_web


def main():
    summary = summarize_pdf_on_web(str(input("Enter arXiv URL of the PDF file: ")))
    with open("pdfs/summary.txt", "w", encoding="utf-8-sig") as f:
        f.write(summary)
    print("Summary created successfully.")


if __name__ == "__main__":
    main()
