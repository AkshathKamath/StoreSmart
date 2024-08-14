import React from "react";
import "./App.css";
import { Routes, Route } from "react-router-dom";
import HomePage from "./views/Home";
import AnalyticsForm from "./views/AnanlyticsForm";
import Timeframe from "./views/TimeFrame";
import StoreWise from "./views/StoreWise";

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/analytics/form" element={<AnalyticsForm />} />
        <Route path="/analytics/timeframe" element={<Timeframe />} />
        <Route path="/analytics/storewise" element={<StoreWise />} />
      </Routes>
    </div>
  );
}

export default App;
