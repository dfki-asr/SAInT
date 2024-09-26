window.addEventListener('beforeunload', function (e) {
    var confirmationMessage = 'You have unsaved data. Do you really want to leave?';
    e.preventDefault();
    e.returnValue = confirmationMessage; // Modern browsers
    return confirmationMessage; // For older browsers
});
