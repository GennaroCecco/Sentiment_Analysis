function inviaDati() {
    var sceltaSelezionata = $('input[name=scelta]:checked').val();
    var input = document.getElementById('fileInput');
    if (input.files.length > 0) {
        var file = input.files[0];
        var formData = new FormData();
        formData.append('scelta', sceltaSelezionata);
        formData.append('file', file);
        $.ajax({
            url: '/ricevi-dati',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                console.log(response)
                $('#corpoRNN').append('<h1 id="risultatoTitolo">Risultato</h1>');
                $('#corpoRNN').css({
                    'margin-top': '20rem',
                    'text-align': 'center'
                });
                $('#risultatoTitolo').css('margin-top', '20px');
                $('#risultatoTitolo').append('<p>' + response + '</p>');
                $('#corpoRNN').append('<button id="restart">Restart</button>');
                $('#restart').on('click', function () {
                    location.reload();
                });
            },
            error: function (xhr, status, error) {
                console.error('Errore durante l\'invio dei dati:', status, error);
            }
        });
    } else {
        console.log('Nessun file selezionato.');
        $('#corpoRNN').append('<h1 id="risultatoTitolo">Risultato</h1>');
        $('#risultatoTitolo').append('<p>' + 'Nessun file selezionato' + '</p>');
        $('#corpoRNN').append('<button id="restart">Restart</button>');
        $('#restart').on('click', function () {
            location.reload();
        });
    }
}

//funzione che fa apparire il footer allo scroll del mouse
document.addEventListener("DOMContentLoaded", function () {
    var footer = document.querySelector("footer");
    var corpoRNN = document.getElementById("corpoRNN");

    // Nascondi il footer all'inizio
    footer.style.opacity = "0";
    footer.style.visibility = "hidden";

    window.addEventListener("load", function () {
        // Quando la pagina e tutte le risorse sono caricate
        var showFooterHeight = corpoRNN.offsetTop + corpoRNN.offsetHeight - window.innerHeight;

        // Aggiorna l'opacità e la visibilità in base alla posizione di scorrimento
        function updateFooterVisibility() {
            var scrollPosition = window.scrollY || window.pageYOffset || document.documentElement.scrollTop;

            if (scrollPosition > showFooterHeight) {
                footer.style.opacity = "1";
                footer.style.visibility = "visible";
            } else {
                footer.style.opacity = "0";
                footer.style.visibility = "hidden";
            }
        }

        // Aggiorna la visibilità del footer quando si verifica lo scroll
        window.addEventListener("scroll", updateFooterVisibility);

        // Chiamata iniziale per gestire lo stato all'inizio
        updateFooterVisibility();
    });
});


// funzione che mostra o nasconde la freccia durante lo scrolling
window.onscroll = function () {
    showScrollToTop();
};

function showScrollToTop() {
    var scrollToTop = document.getElementById("scrollToTop");
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        scrollToTop.style.display = "block";
    } else {
        scrollToTop.style.display = "none";
    }
}

// Torna all'inizio con transizione più lenta quando clicchi sulla freccia
document.getElementById("scrollToTop").addEventListener("click", function () {
    scrollToTopWithTransition();
});

function scrollToTopWithTransition() {
    var scrollOptions = {
        top: 0,
        behavior: "smooth",
    };

    window.scrollTo(scrollOptions);
}















