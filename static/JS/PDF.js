function generatePDF() {
    console.log("printing PDF");
    const { jsPDF } = window.jspdf;
    const currentDate = new Date();
    const formattedDate = `${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, '0')}-${String(currentDate.getDate()).padStart(2, '0')}`;

    const text1 = "Here is your car valuation for the next 5 years:";
    const text2 = "Thank you for using our services!";

    // Step 1: Load the text and image first
    Promise.all([
        fetch("../static/txt/predictedPrice.txt").then(r => r.text()),
        new Promise((resolve) => {
            const img = new Image();
            img.src = "../static/plot/valuation.png";
            img.onload = () => resolve(img);
        })
    ]).then(([predictedText, img]) => {
        const text3 = `Your total depreciation would be: ${predictedText.trim()}`;
        const doc = new jsPDF();
        const pageWidth = doc.internal.pageSize.width;

        const x1 = (pageWidth - doc.getTextWidth(text1)) / 2;
        const x2 = (pageWidth - doc.getTextWidth(text2)) / 2;
        const x3 = (pageWidth - doc.getTextWidth(text3)) / 2;

        const y1 = 100;
        const y2 = y1 + 120;
        const y3 = y2 + 20;

        doc.text(text1, x1, y1);
        doc.addImage(img, 'PNG', 15, y1 + 10, 180, 100); // Adjust size if needed
        doc.text(text2, x2, y2);
        doc.text(text3, x3, y3);

        doc.save(`carValuation~${formattedDate}.pdf`);
    });
}