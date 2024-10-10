import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";
import NavbarComponent from "../components/Navbar";
import Dropdown from "../components/Dropdown";

const StoreWise = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(
          "https://storesmart-backend-production.up.railway.app/analytics/storewise"
        );
        // const response = await axios.get(
        //   "http://localhost:8000/analytics/storewise"
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
        <h5>Mission Hill</h5>
        <table className="table table-hover table-bordered">
          <thead>
            <tr>
              <th scope="col">S No.</th>
              <th scope="col">Product Category</th>
              <th scope="col">Quantity</th>
              <th scope="col">Cost of Goods($)</th>
              <th scope="col">Total Revenue($)</th>
              <th scope="col">Net Profit($)</th>
            </tr>
          </thead>
          <tbody className="table-group-divider">
            {data.list1.map((obj, index) => (
              <tr>
                <th scope="row"> {index + 1} </th>
                <td> {obj.ProductLine} </td>
                <td> {obj.Quantity} </td>
                <td> {obj.cogs} </td>
                <td> {obj.TotalAmountWithTax} </td>
                <td> {obj.NetProfit} </td>
              </tr>
            ))}
          </tbody>
        </table>
        <img
          src="https://storesmart.s3.amazonaws.com/storewise_1.png"
          alt="Unable"
        />
        <br />
        <br />
        <h5>Fenway</h5>
        <table className="table table-hover table-bordered">
          <thead>
            <tr>
              <th scope="col">S No.</th>
              <th scope="col">Product Category</th>
              <th scope="col">Quantity</th>
              <th scope="col">Cost of Goods($)</th>
              <th scope="col">Total Revenue($)</th>
              <th scope="col">Net Profit($)</th>
            </tr>
          </thead>
          <tbody className="table-group-divider">
            {data.list2.map((obj, index) => (
              <tr>
                <th scope="row"> {index + 1} </th>
                <td> {obj.ProductLine} </td>
                <td> {obj.Quantity} </td>
                <td> {obj.cogs} </td>
                <td> {obj.TotalAmountWithTax} </td>
                <td> {obj.NetProfit} </td>
              </tr>
            ))}
          </tbody>
        </table>
        <img
          src="https://storesmart.s3.amazonaws.com/storewise_2.png"
          alt="Unable"
        />
        <br />
        <br />
        <h5>Roxbury</h5>
        <table className="table table-hover table-bordered">
          <thead>
            <tr>
              <th scope="col">S No.</th>
              <th scope="col">Product Category</th>
              <th scope="col">Quantity</th>
              <th scope="col">Cost of Goods($)</th>
              <th scope="col">Total Revenue($)</th>
              <th scope="col">Net Profit($)</th>
            </tr>
          </thead>
          <tbody className="table-group-divider">
            {data.list3.map((obj, index) => (
              <tr>
                <th scope="row"> {index + 1} </th>
                <td> {obj.ProductLine} </td>
                <td> {obj.Quantity} </td>
                <td> {obj.cogs} </td>
                <td> {obj.TotalAmountWithTax} </td>
                <td> {obj.NetProfit} </td>
              </tr>
            ))}
          </tbody>
        </table>
        <img
          src="https://storesmart.s3.amazonaws.com/storewise_3.png"
          alt="Unable"
        />
        <br />
        <br />
      </div>
    </div>
  );
};

export default StoreWise;
