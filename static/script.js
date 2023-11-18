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
