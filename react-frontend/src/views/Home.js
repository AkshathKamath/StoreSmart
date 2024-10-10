import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";
import NavbarComponent from "../components/Navbar";

const HomePage = () => {
  const [data, setData] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);

  //For the 'View Analytics' Button, get API request to node backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(
          "https://storesmart-backend-production.up.railway.app/home"
        );
        // const response = await axios.get("http://localhost:8000/home");
        // const response = await axios.get("api/home");
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, []);

  let Content;
  if (data.msg > 0) {
    Content = (
      <div className="container mt-4">
        <h6>Or view past data analytics:</h6>
        <a href="/analytics/form" className="btn btn-primary">
          View Analytics
        </a>
      </div>
    );
  } else {
    Content = "";
  }

  //To handle form-upload, post api request to backend
  const handleFileChange = (event) => {
    // console.log(event.target.files[0]);
    const file = event.target.files[0];
    setSelectedFile(file);
    console.log(selectedFile);
  };
  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("file", selectedFile);
    // console.log(formData);

    try {
      console.log(formData);
      const response = await axios.post(
        "https://storesmart-backend-production.up.railway.app/analytics/form",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      //----------------------------------------------
      // const response = await axios.post(
      //   "http://http://localhost:8000/analytics/form",
      //   formData,
      //   {
      //     headers: {
      //       "Content-Type": "multipart/form-data",
      //     },
      //   }
      // );
      // const response = await axios.post("api/analytics/form", formData, {
      //   headers: {
      //     "Content-Type": "multipart/form-data",
      //   },
      // });
      console.log("File uploaded successfully:", response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  return (
    <div>
      <NavbarComponent />
      <div className="container mt-4">
        <figure className="text-center">
          <blockquote className="blockquote">
            <p align="center">
              StoreSmart empowers local supermarkets to leverage data-driven
              insights and machine learning capabilities to stay competitive,
              enhance customer satisfaction, and drive business growth in the
              dynamic retail landscape.
            </p>
          </blockquote>
        </figure>
      </div>
      <div className="container mt-4">
        <form onSubmit={handleSubmit} encType="multipart/form-data">
          <div className="mb-3">
            <label htmlFor="file" className="form-label">
              Upload your store's new dataset:
            </label>
            <input
              type="file"
              id="file"
              name="file"
              className="form-control"
              onChange={handleFileChange}
            />
          </div>
          <a type="submit" className="btn btn-primary" href="/analytics/form">
            Upload
          </a>
        </form>
      </div>
      {Content}
      <div className="container mt-5">
        <h6>Features and advantages</h6>
        <div className="list-group">
          <p className="list-group-item list-group-item-action flex-column align-items-start">
            <div className="d-flex w-100 justify-content-between">
              <b className="mb-1">Increased Revenue and Sales</b>
            </div>
            <p className="mb-1">
              By optimizing inventory, supermarkets can increase sales by up to
              15%, leveraging data-driven insights to ensure popular items are
              always stocked while minimizing waste and maximizing
              profitability.
            </p>
          </p>
          <p className="list-group-item list-group-item-action flex-column align-items-start">
            <div className="d-flex w-100 justify-content-between">
              <b className="mb-1">Operational Efficiency and Cost Savings</b>
            </div>
            <p className="mb-1">
              Streamline operations with automated analytics, saving up to 10%
              in staff time by efficiently managing inventory, predicting
              customer demand, and automating routine tasks, allowing staff to
              focus on delivering exceptional service and strategic initiatives.
            </p>
          </p>
          <p className="list-group-item list-group-item-action flex-column align-items-start">
            <div className="d-flex w-100 justify-content-between">
              <b className="mb-1">Sales Forecasting</b>
            </div>
            <p className="mb-1">
              Predict future sales trends based on historical data, helping
              supermarkets optimize inventory and staffing by anticipating
              seasonal fluctuations, identifying emerging product trends, and
              ensuring the right mix of products are available to meet customer
              demand efficiently.
            </p>
          </p>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
