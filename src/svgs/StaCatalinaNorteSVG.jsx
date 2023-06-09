import React from "react";
import { Context } from "../Context";

const StaCatalinaNorteSVG = () => {
	const { map, setHeatMap, setHeatMapItems } = React.useContext(Context);
	const ref = React.useRef(null);
	React.useEffect(() => {
		setHeatMap(ref.current);
		setHeatMapItems(ref.current.children);
	}, [ref, map, setHeatMap, setHeatMapItems]);
	return (
		<svg
			ref={ref}
			width='1787'
			height='832'
			viewBox='0 0 1787 832'
			id='sta_catalina_norte'
			className='mask'
			fill='none'
			xmlns='http://www.w3.org/2000/svg'>
			<path id="Rectangle 2" d="M596 39.5L447 50.0722V50H54.8682L17.1149 64.6066L0 108.426V445H298V444H596V39.5Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 2</textPath></text>
			</path>
			<path id="Rectangle 1" d="M298 444H596V764.18L525.527 777.829L447 793.5V793.207L371.493 814.875L298 831V832H0V445H298V444Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 1</textPath></text>
			</path >
			<path id="Rectangle 4" d="M1192 4.5L1043 11.0483V11H894V12L745 22.582V22L596 38.6398V399H894V398H1192V4.5Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 4</textPath></text>
			</path >
			<path id="Rectangle 3" d="M894 398H1192V722H1043V722.917L894 729V730.869L745 742V743.303L596 764.5V399H894V398Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 3</textPath></text>
			</path >
			<path id="Rectangle 5" d="M1490 0H1639L1710.39 4.03125L1719 129H1717.87L1730 258H1733.31L1755 387H1490V388H1192V6.54297L1341 1H1490V0Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 5</textPath></text>
			</path >
			<path id="Rectangle 6" d="M1490 387H1753.88L1770 516H1772.9L1788 645L1754.78 681.046L1708.47 703.892L1639 711H1490V712H1192V388H1490V387Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 6</textPath></text>
			</path >

		</svg >
	);
};

export default StaCatalinaNorteSVG;
