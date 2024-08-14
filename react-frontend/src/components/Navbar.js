import React from "react";
import {
  Navbar,
  Nav,
  Container,
  NavItem,
  NavLink,
  NavbarBrand,
  NavbarToggle,
  NavbarCollapse,
} from "react-bootstrap";

const NavbarComponent = () => {
  return (
    <Navbar bg="primary" variant="dark" expand="default">
      <Container fluid>
        <NavbarBrand href="/home">
          StoreSmart: Empowering local supermarkets with analytics and ML!
        </NavbarBrand>
        <NavbarToggle aria-controls="navbarSupportedContent" />
        <NavbarCollapse id="navbarSupportedContent">
          <Nav className="me-auto mb-2 mb-lg-0 px-5">
            <NavItem>
              <NavLink href="/home">Home</NavLink>
            </NavItem>
            <NavItem>
              <NavLink href="/analytics/form">Analytics</NavLink>
            </NavItem>
            <NavItem>
              <NavLink href="/model">ML</NavLink>
            </NavItem>
          </Nav>
        </NavbarCollapse>
      </Container>
    </Navbar>
  );
};

export default NavbarComponent;
