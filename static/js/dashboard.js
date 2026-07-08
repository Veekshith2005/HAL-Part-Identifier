function toggleSidebar()
{
    document.getElementById("sidebar")
            .classList.toggle("collapsed");
}

function showUpload()
{
    document.getElementById("upload-section").style.display = "block";

    const webcam = document.getElementById("webcam-section");

    if(webcam)
    {
        webcam.style.display = "none";
    }
}

function showWebcam()
{
    document.getElementById("upload-section").style.display = "none";

    const webcam = document.getElementById("webcam-section");

    if(webcam)
    {
        webcam.style.display = "block";
    }
}

function toggleDarkMode()
{
    document.body.classList.toggle("dark");
}

async function updateDetection()
{
    try
    {
        const response = await fetch('/detection_data');
        const data = await response.json();

        document.getElementById("part_number").innerText = data.part_number;
        document.getElementById("part_name").innerText = data.part_name;
        document.getElementById("issue").innerText = data.issue;
    }
    catch(err)
    {
        console.log(err);
    }
}

setInterval(updateDetection, 1000);

window.onload = function()
{
    showUpload();
}