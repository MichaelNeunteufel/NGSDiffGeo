(function () {
  "use strict";

  var script = document.currentScript;
  if (!script) return;
  var buildRoot = new URL("../", script.src);
  var parts = buildRoot.pathname.split("/").filter(Boolean);
  var channel = parts[parts.length - 1] || "";
  var isVersion = channel === "dev" || /^v\d+\.\d+\.\d+$/.test(channel);
  var siteRoot = isVersion ? new URL("../", buildRoot) : buildRoot;
  var currentPath = isVersion ? channel + "/" : "";

  function whenReady(callback) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
    } else {
      callback();
    }
  }

  fetch(new URL("versions.json", siteRoot))
    .then(function (response) {
      if (!response.ok) throw new Error("No version manifest");
      return response.json();
    })
    .then(function (manifest) {
      whenReady(function () {
        var entries = manifest.versions || [];
        if (entries.length < 2) return;
        var container = document.querySelector(".wy-side-nav-search");
        if (!container) return;

        var label = document.createElement("label");
        label.className = "ngsdiffgeo-version-label";
        label.textContent = "Documentation version";
        var select = document.createElement("select");
        select.className = "ngsdiffgeo-version-select";
        select.setAttribute("aria-label", "Documentation version");
        entries.forEach(function (entry) {
          var option = document.createElement("option");
          option.value = entry.path;
          option.textContent = entry.label;
          option.selected = entry.path === currentPath;
          select.appendChild(option);
        });
        var page = window.location.pathname.slice(buildRoot.pathname.length) || "index.html";
        select.addEventListener("change", function () {
          var target = entries.find(function (entry) { return entry.path === select.value; });
          if (!target) return;
          var samePage = target.pages.includes(page);
          var url = new URL(target.path + (samePage ? page : "index.html"), siteRoot);
          if (samePage) url.hash = window.location.hash;
          window.location.assign(url.href);
        });
        label.appendChild(select);
        container.appendChild(label);
      });
    })
    .catch(function () {
      // Standalone Sphinx builds have no versions.json and need no selector.
    });
})();
