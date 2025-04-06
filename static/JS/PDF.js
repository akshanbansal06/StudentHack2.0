function generatePDF() {
    const currentDate = new Date();
    const year = currentDate.getFullYear();
    const month = String(currentDate.getMonth() + 1).padStart(2, '0'); // Months are zero-based
    const day = String(currentDate.getDate()).padStart(2, '0');
    const formattedDate = `${year}-${month}-${day}`;

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const text1 = "Here is your car valuation for the next 5 years:";
    const text2 = "Thank you for using our services!";
    const text3 = "Your total depreciation would be:";

    const img = new Image();
    img.src = "path_to_your_image.png";


    const pageWidth = doc.internal.pageSize.width; // Get the page width
    const textWidth1 = doc.getTextWidth(text1);    // Calculate width of first text
    const textWidth2 = doc.getTextWidth(text2);  
    const textWidth3 = doc.getTextWidth(text3);  // Calculate width of second text
    const xPosition1 = (pageWidth / 2) - (textWidth1 / 2); // Center horizontally
    const xPosition2 = (pageWidth / 2) - (textWidth2 / 2);
    const xPosition3 = (pageWidth / 2) - (textWidth3 / 2); // Center horizontally
    const yPosition1 = 100; // First line's y-coordinate
    const yPosition2 = yPosition1 + 20 + img.height; // Add space; second line appears 20 units below
    const yPosition3 = yPosition2 + 20

    const imgXPosition = (pageWidth / 2) - (img.width / 2); // Center horizontally
    const imgYPosition = (pageWidth / 2) - (img.width / 2); // Center horizontall

    // Add the first line of text
    doc.text(text1, xPosition1, yPosition1);
    doc.addImage(base64Image, 'PNG', 15, 40, img.width, img.height)
    doc.text(text2, xPosition2, yPosition2);
    doc.text(text3, xPosition3, yPosition3);



    doc.text("Here is your car valuation for the next 5 years: ", 100, 100);
    doc.save(`carValuation~${formattedDate}~.pdf`);                
}