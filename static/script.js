const button = document.getElementById("summarize-btn");

const input = document.getElementById("dialogue-input");

const loading = document.getElementById("loading");

const result = document.getElementById("result");

const summaryText = document.getElementById("summary-text");


button.addEventListener("click", async () => {

    const dialogue = input.value.trim();


    if (!dialogue) {

        alert("Please enter some text.");

        return;

    }


    button.disabled = true;

    loading.style.display = "block";

    result.style.display = "none";


    try {

        const response = await fetch("/summarize/", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                dialogue: dialogue

            })

        });


        if (!response.ok) {

            throw new Error(
                `Server error: ${response.status}`
            );

        }


        const data = await response.json();


        summaryText.textContent = data.summary;

        result.style.display = "block";


    } catch (error) {

        alert(
            "Something went wrong: " + error.message
        );

    } finally {

        button.disabled = false;

        loading.style.display = "none";

    }

});