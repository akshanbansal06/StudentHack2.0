import React from "react";
import {Link} from "react-router-dom";

// NavBar component 
const NavBar: React.FC = () => {
    return (
        <nav className="bg-blue0 text-white px-6 py-2 shadow-md fixed top-0 w-full z-50">
            <div className="container mx-auto flex items-center justify-between">
            <div className="text-2xl font-bold text-blue2 font-Font">VALCAR</div>
                <ul className="flex space-x-6">
                    <li><Link to="/" className="hover:text-blue2 transition font-Font">Home</Link></li>
                    <li><Link to="/valuation" className="hover:text-blue2 transition font-Font">Valuation</Link></li>
                </ul>
            </div>
        </nav>

    );
};

export default NavBar;
