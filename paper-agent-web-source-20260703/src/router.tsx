import { Navigate, createBrowserRouter } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { EvaluationPage } from "./pages/EvaluationPage";
import { GraphPage } from "./pages/GraphPage";
import { LogsPage } from "./pages/LogsPage";
import { ResultsPage } from "./pages/ResultsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { WorkbenchPage } from "./pages/WorkbenchPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <WorkbenchPage /> },
      { path: "results/:jobId", element: <ResultsPage /> },
      { path: "graph/:jobId", element: <GraphPage /> },
      { path: "eval", element: <EvaluationPage /> },
      { path: "logs/:jobId", element: <LogsPage /> },
      { path: "settings", element: <SettingsPage /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);
