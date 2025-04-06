function generatePDF() {
    const currentDate = new Date();
    const year = currentDate.getFullYear();
    const month = String(currentDate.getMonth() + 1).padStart(2, '0');
    const day = String(currentDate.getDate()).padStart(2, '0');
    const formattedDate = `${year}-${month}-${day}`;

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const text1 = "Here is your car valuation for the next 5 years:";
    const text2 = "Thank you for using our services!";
    const text3 = "Your total depreciation would be:";

    const img = new Image();
    img.src = "path_to_your_image.png"; // Update with actual image path

    img.onload = function () {
        const pageWidth = doc.internal.pageSize.width;
        const textWidth1 = doc.getTextWidth(text1);
        const textWidth2 = doc.getTextWidth(text2);
        const textWidth3 = doc.getTextWidth(text3);
        const xPosition1 = (pageWidth / 2) - (textWidth1 / 2);
        const xPosition2 = (pageWidth / 2) - (textWidth2 / 2);
        const xPosition3 = (pageWidth / 2) - (textWidth3 / 2);
        const yPosition1 = 100;
        const yPosition2 = yPosition1 + 20 + 100; // Adjust based on image height
        const yPosition3 = yPosition2 + 20;

        const imgXPosition = 15; // Example x position
        const imgYPosition = yPosition1 + 10; // Example y position

        // Add the first line of text
        doc.text(text1, xPosition1, yPosition1);

        // Add the image
        doc.addImage(img, 'PNG', imgXPosition, imgYPosition, 180, 100); // Example dimensions

        // Add subsequent text
        doc.text(text2, xPosition2, yPosition2);
        doc.text(text3, xPosition3, yPosition3);

        // Save the PDF
        doc.save(`carValuation~${formattedDate}~.pdf`);
    };
}