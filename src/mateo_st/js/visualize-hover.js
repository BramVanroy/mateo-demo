function enter(cls) {
  const els = window.parent.document.querySelectorAll(`.${cls}`);

  els.forEach(el => {el.classList.add("hover")})
}
function leave(cls) {
  const els = window.parent.document.querySelectorAll(`.${cls}`);
  els.forEach(el => {el.classList.remove("hover")})
}
