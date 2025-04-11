import React from "react";
import { Link } from "react-router-dom";

// Footer Component
const Footer: React.FC = () => {
  return (
    <footer className="bg-blue0 text-white px-6 py-10 shadow-md">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Column 1 */}
        <div>
          <h1 className="text-2xl font-bold">Our Website</h1>
          <p className="mt-2 text-gray-300">
            Giving you the best experience and accurate car valuations today.
          </p>
        </div>

        {/* Column 2 */}
        <div>
          <h2 className="text-lg font-bold mb-4">Quick Links</h2>
          {/* ul for unordered lists, li is list item */}
          <ul className="space-y-2">
            <li>
              <Link to="/" className="hover:text-gray-400">
                Home
              </Link>
            </li>
            <li>
              <Link to="/about" className="hover:text-gray-400">
                About Us
              </Link>
            </li>
            <li>
              <Link to="/contact" className="hover:text-gray-400">
                Contact
              </Link>
            </li>
          </ul>
        </div>

        {/* Column 3 */}
        <div>
          <h2 className="text-lg font-bold mb-4">Follow Us</h2>
          <ul className="space-y-2">
            <li>
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-gray-400"
              >
                X (formerly Twitter)
              </a>
            </li>
            <li>
              <a
                href="https://facebook.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-gray-400"
              >
                Facebook
              </a>
            </li>
            <li>
              <a
                href="https://instagram.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-gray-400"
              >
                Instagram
              </a>
            </li>
          </ul>
        </div>
      </div>

      {/* Bottom Section (Copyright) */}
      <div className="mt-8 border-t border-gray-400 pt-4 text-center text-gray-300">
        © 2025 Valcar. All rights reserved.
      </div>
    </footer>
  );
};

export default Footer;