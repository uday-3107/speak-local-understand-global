import { initLectureView } from "./lecture.js";
import { initHistoryView } from "./history.js";
import { initDashboardView } from "./dashboard.js";

const navButtons = Array.from(document.querySelectorAll(".nav-btn"));
const view = document.querySelector("#view");

let activeCleanup = null;

const views = {
  lecture: (root) => initLectureView(root),
  history: (root) => initHistoryView(root),
  dashboard: (root) => initDashboardView(root),
};

function navigate(page) {
  if (activeCleanup) {
    activeCleanup();
    activeCleanup = null;
  }
  navButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.page === page));
  view.replaceChildren();
  const result = views[page](view);
  if (result && typeof result.cleanup === "function") {
    activeCleanup = result.cleanup;
  }
}

navButtons.forEach((btn) => btn.addEventListener("click", () => navigate(btn.dataset.page)));

navigate("lecture");