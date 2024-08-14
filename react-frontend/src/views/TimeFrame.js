import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";
import NavbarComponent from "../components/Navbar";
import Dropdown from "../components/Dropdown";

const Timeframe = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(
          "http://44.216.20.188:8000/analytics/timeframe"
        );
        // const response = await axios.get(
        //   "http://localhost:8000/analytics/timeframe"
        // );
        setData(response.data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching data:", error);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <p>Loading...</p>;

  return (
    <div>
      <NavbarComponent />
      <br />
      <div className="container mt-4">
        <Dropdown />
      </div>
      <br />
      <div className="container ms-4" align="left">
        <h5>Month-Wise breakdown</h5>
        <table className="table table-hover table-bordered">
          <thead>
            <tr>
              <th scope="col">S No.</th>
              <th scope="col">Month</th>
              <th scope="col">Quantity</th>
              <th scope="col">Tax($)</th>
              <th scope="col">Cost of Goods($)</th>
            </tr>
          </thead>
          <tbody className="table-group-divider">
            {data.list1.map((obj, index) => (
              <tr>
                <th scope="row"> {index + 1} </th>
                <td> {obj.Month} </td>
                <td> {obj.Quantity} </td>
                <td> {obj.Tax} </td>
                <td> {obj.cogs} </td>
              </tr>
            ))}
          </tbody>
        </table>
        <img
          src="https://storesmart.s3.amazonaws.com/timeframe_1.png"
          alt="Unable"
        />
        <br />
        <br />
        <h5>General Weekly Analysis</h5>
        <table className="table table-hover table-bordered">
          <thead>
            <tr>
              <th scope="col">S No.</th>
              <th scope="col">Week of Month</th>
              <th scope="col">Quantity</th>
              <th scope="col">Average Revenue($)</th>
              <th scope="col">Average Gross Profit($)</th>
              <th scope="col">Customer Rating</th>
            </tr>
          </thead>
          <tbody className="table-group-divider">
            {data.list2.map((obj, index) => (
              <tr>
                <th scope="row"> {index + 1} </th>
                <td> {obj.WoM} </td>
                <td> {obj.Quantity} </td>
                <td> {obj.TotalAmountWithTax} </td>
                <td> {obj.GrossProfit} </td>
                <td> {obj.CustomerRating} </td>
              </tr>
            ))}
          </tbody>
        </table>
        <br />
        <h5>General Daily Analysis</h5>
        <table className="table table-hover table-bordered">
          <thead>
            <tr>
              <th scope="col">S No.</th>
              <th scope="col">Time Slot</th>
              <th scope="col">Quantity</th>
              <th scope="col">Average Revenue($)</th>
              <th scope="col">Average Gross Profit($)</th>
              <th scope="col">Customer Rating</th>
            </tr>
          </thead>
          <tbody className="table-group-divider">
            {data.list3.map((obj, index) => (
              <tr>
                <th scope="row"> {index + 1} </th>
                <td> {obj.Slot} </td>
                <td> {obj.Quantity} </td>
                <td> {obj.TotalAmountWithTax} </td>
                <td> {obj.GrossProfit} </td>
                <td> {obj.CustomerRating} </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Timeframe;
