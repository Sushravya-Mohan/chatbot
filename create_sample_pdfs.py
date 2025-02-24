import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_pdf(file_path, text):
    """Creates a PDF file with the given text."""
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 40)
    text_object.setFont("Helvetica", 12)
    for line in text.splitlines():
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()

def main():
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)
    
    # Define sample content for PDFs
    faq_content = (
        "FAQ Data\n"
        "This is a simulated FAQ document.\n"
        "It contains multiple lines of text.\n"
        "FAQ: Where is my order?\n"
        "Answer: Your order is on the way, and it will arrive within 3-5 business days.\n"
        "FAQ: How do I return an item?\n"
        "Answer: To return an item, please visit our returns page."
    )
    
    conversation_content = (
        "Historical Conversations\n"
        "Simulated historical conversation data.\n"
        "Customer: I haven't received my order.\n"
        "Support: Please check your email for shipping updates.\n"
        "Customer: My package is delayed, can you help?\n"
        "Support: Sure, let me look into it for you."
    )
    
    # Create the PDFs
    create_sample_pdf("data/faq_data.pdf", faq_content)
    create_sample_pdf("data/historical_conversations.pdf", conversation_content)
    print("Sample PDFs created in the 'data' folder.")

if __name__ == "__main__":
    main()
