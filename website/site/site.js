/* robotmsgs.com: theme, drawer, sidebar filter, copy buttons, on-this-page highlight. No dependencies. */
(function () {
  var root = document.documentElement;
  var body = document.body;

  // theme
  var themeBtn = document.querySelector(".theme-btn");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) {}
    });
  }

  // drawer (phones)
  var menuBtn = document.querySelector(".menu-btn");
  var scrim = document.querySelector(".scrim");
  function setMenu(open) {
    body.classList.toggle("menu-open", open);
    if (menuBtn) {
      menuBtn.setAttribute("aria-expanded", String(open));
      menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    }
    if (scrim) scrim.hidden = !open;
  }
  if (menuBtn) menuBtn.addEventListener("click", function () { setMenu(!body.classList.contains("menu-open")); });
  if (scrim) scrim.addEventListener("click", function () { setMenu(false); });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") setMenu(false);
  });
  window.addEventListener("resize", function () { if (window.innerWidth > 767) setMenu(false); });

  // sidebar filter
  var search = document.getElementById("search");
  if (search) {
    var groups = document.querySelectorAll("[data-group]");
    var empty = document.querySelector(".sb-empty");
    var filter = function () {
      var q = search.value.trim().toLowerCase();
      var any = false;
      groups.forEach(function (group) {
        var shown = 0;
        group.querySelectorAll("li").forEach(function (item) {
          var hit = !q || item.dataset.name.indexOf(q) !== -1 || group.querySelector(".sb-h").textContent.indexOf(q) !== -1;
          item.hidden = !hit;
          if (hit) shown++;
        });
        group.hidden = shown === 0;
        if (shown) any = true;
      });
      if (empty) empty.hidden = any;
    };
    search.addEventListener("input", filter);
    search.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        var first = document.querySelector("[data-group]:not([hidden]) li:not([hidden]) a");
        if (first) window.location.href = first.href;
      }
    });
    document.addEventListener("keydown", function (event) {
      var tag = (event.target.tagName || "").toLowerCase();
      if (event.key === "/" && tag !== "input" && tag !== "textarea") {
        event.preventDefault();
        if (window.innerWidth <= 767) setMenu(true);
        search.focus();
      }
    });
    // keep the active type in view inside the sidebar
    var active = document.querySelector('.sidebar [aria-current="page"]');
    var sidebar = document.getElementById("sidebar");
    if (active && sidebar) sidebar.scrollTop = Math.max(0, active.offsetTop - sidebar.clientHeight / 2);
  }

  // copy buttons
  document.querySelectorAll(".copy").forEach(function (button) {
    button.addEventListener("click", function () {
      var text = button.parentElement.querySelector("pre").innerText.replace(/\n$/, "");
      var done = function () {
        button.textContent = "Copied";
        button.classList.add("done");
        setTimeout(function () { button.textContent = "Copy"; button.classList.remove("done"); }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(text, done); });
      } else {
        fallback(text, done);
      }
    });
  });
  function fallback(text, done) {
    var area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); done(); } catch (e) {}
    document.body.removeChild(area);
  }

  // on this page
  var links = document.querySelectorAll(".toc a");
  if (links.length && "IntersectionObserver" in window) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          links.forEach(function (a) { a.classList.remove("active"); });
          var a = byId[entry.target.id];
          if (a) a.classList.add("active");
        }
      });
    }, { rootMargin: "-70px 0px -70% 0px" });
    Object.keys(byId).forEach(function (id) {
      var target = document.getElementById(id);
      if (target) observer.observe(target);
    });
  }
})();
