import { Typography } from "@mui/material";
import { PageContainer } from "@toolpad/core";

export default function SearchPage() {
  return (    
    <PageContainer breadcrumbs={[]}>
      <Typography>
        Thank you to Comcast and Akel Homes for their generous donations to the Education Foundation and Project 001.
        <br />
        <br />
        <strong> Comcast donates $35,000 to Project 001 </strong>
        <br />
        Thanks to Comcast, Project 001 was able to secure laptops for each student to use for Project 001.  In addition,
        we were able to buy a workstation with two GPUs to process the videos in near real-time.
        <br />
        <br />
        <strong> AKEL Homes donates $10,000 to Project 001 </strong>
        Thanks to a generous donation from <a href="https://akelhomes.com/">
		AKEL Homes </a> the Education Foundation was able to purchase an additional laptop and an electronic
		whiteboard from <a href="vibe.us" Vibe> </a>
        <br />


      </Typography>
    </PageContainer>
  );
}
