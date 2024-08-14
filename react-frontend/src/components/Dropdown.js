import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { useNavigate } from "react-router-dom";

const Dropdown = () => {
  const [selectedOption, setSelectedOption] = useState("form");
  const dropdownOptions = [
    { value: "form", label: "General Overview" },
    { value: "storewise", label: "Store-Wise Product Analysis" },
    { value: "timeframe", label: "Timeframe Analysis" },
  ];
  const navigate = useNavigate();

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
  };

  const handleSubmitOption = (event) => {
    event.preventDefault();
    // console.log("Selected option:", selectedOption);
    // Handle the selected option (e.g., send it to the server or use it in your logic)
    navigate(`/analytics/${selectedOption}`);
  };

  return (
    <form onSubmit={handleSubmitOption}>
      <div className="mb-3">
        <label htmlFor="dropdown" className="form-label">
          Select analysis to view:
        </label>
        <select
          id="dropdown"
          value={selectedOption}
          onChange={handleOptionChange}
          className="form-select"
        >
          {dropdownOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
      <button type="submit" className="btn btn-primary">
        Submit
      </button>
    </form>
  );
};

export default Dropdown;
