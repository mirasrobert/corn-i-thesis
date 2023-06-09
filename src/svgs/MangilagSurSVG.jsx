import React from "react";
import { Context } from "../Context";

const MangilagSurSVG = () => {
	const { map, setHeatMap, setHeatMapItems } = React.useContext(Context);
	const ref = React.useRef(null);
	React.useEffect(() => {
		setHeatMap(ref.current);
		setHeatMapItems(ref.current.children);
	}, [ref, map, setHeatMap, setHeatMapItems]);
	return (
		<svg
			ref={ref}
			width='1632'
			height='1125'
			viewBox='0 0 1632 1125'
			id='mangilag_sur'
			className='mask'
			fill='none'
			xmlns='http://www.w3.org/2000/svg'>
			<path id="Rectangle 1" fillRule="evenodd" clipRule="evenodd" d="M48.7105 457H666V618H665V940H663.932V1106.97L630.237 1118.03L581.957 1126.08L559.326 1103.45L503 1100.93V940H502.938V1100.94L496.903 1108.99L478.798 1111L466.224 1123.57H441.581V1111H420.96L414.422 1100.94H342V940H341.947V1101.95L327.864 1115.03L181 1120.06V1120H133.741L95.0265 1100.89L7.54178 1098.88L23.6309 1022.96L7.54178 986.759L0.0705339 939.942H0V779H182V778.942H0L12.5736 618H183V617.976H14L48.7105 457ZM182 939.942V940H181L159.999 939.942H182Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 1</textPath></text>
			</path>
			<path id="Rectangle 2" d="M666 457H1310V618H1309V940H1307.97V1108.51L1262.19 1112.03L1147 1100.97V1101H919.084L891.916 1059.24L825 1043.64V940H824.98V1043.13L735.938 1055.2L719.84 1100.98L664 1107.52V940H665V618H666V457Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 2</textPath></text>
			</path >
			<path id="Rectangle 3" d="M1632 457H1310V618H1309V940H1308V1111L1329.12 1121.06L1402.55 1119.05L1468.94 1100.94V940H1469V1100.93L1481.57 1119.04L1515.27 1122.56L1560.03 1100.93L1614.34 1060.7L1632.95 1009.9L1629.93 940H1631V618H1632V457Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 3</textPath></text>
			</path >
			<path id="Rectangle 4" d="M184 15V15.5H133.605L97.3065 32.1367L60 71.4598V134.982H184V135H59.5V231.072L82.254 296.806H83.682L52 457.731H183.406V457H666V135.372H667.845V15H184Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 4</textPath></text>
			</path >
			<path id="Rectangle 5" d="M1310 135L1311.1 134.919V0L1278.88 2.01371L1264.28 15.1028H1150.84V15H667V135H666V457H1310V135Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 5</textPath></text>
			</path >
			<path id="Rectangle 6" d="M1311 0.5H1351.78L1376.95 11.5756L1472.1 19.6305V21.0126L1535.75 29.0958L1614.69 21L1633.92 135.353H1632V457H1310V135L1311 134.917V0.5Z" fill="#D9D9D9">
				<text><textPath className="tooltiptext">Crop Group 6</textPath></text>
			</path >
		</svg >
	);
};

export default MangilagSurSVG;
